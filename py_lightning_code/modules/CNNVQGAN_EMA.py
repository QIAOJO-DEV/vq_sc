from typing import List, Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

import PIL
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms as T
import pytorch_lightning as pl

from .layers import Cnn_Encoder as Encoder, Cnn_Decoder as Decoder
from .quantizers import VectorQuantizer ,EMAQuantizer
from py_lightning_code.utils.general import initialize_from_config
class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
        error_strategy="none",
        error_prob=0.05,
        top_k=500,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = EMAQuantizer(embed_dim, n_embed, error_strategy='none', error_prob=error_prob,decay=decay)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = EMAQuantizer(embed_dim, n_embed, error_strategy='none', error_prob=error_prob,decay=decay)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b
    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        return dec
    def decode_index(self, id_t, id_b):
        quant_t = self.quantize_t.embed_code(id_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(id_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        return self.decode(quant_t, quant_b)
    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff
class CNNVQGAN_EMA(pl.LightningModule):
    def __init__(self, image_key: str, image_size: int, model_param: OmegaConf, 
                 loss: OmegaConf, learning_rate:float=1e-4,epochs:int=500,path: Optional[str] = None) -> None:
        super().__init__()
        self.image_key = image_key
        self.image_size = image_size
        self.model = VQVAE(**model_param)
        self.loss = initialize_from_config(loss)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.path = path
        self.automatic_optimization = False#开启手动优化
    def forward(self, input):
        dec, diff = self.model(input)
        return dec, diff
    def get_input(self, batch: Tuple[Any, Any], key: str = 'image') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()
    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        optim_groups = list(self.model.parameters())
        
        optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        schedulers = []
        warmup_epoches =5
        if hasattr(self.loss, 'discriminator'):
            optimizers.append(torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4))
            
        schedulers = [
                {
                    'scheduler': lr_scheduler.SequentialLR(optimizer, schedulers=[
                        lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epoches),
                        lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs-warmup_epoches,eta_min=1e-6)
                        #lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-3, total_iters=self.epochs-warmup_epoches)
                    ], milestones=[warmup_epoches]),
                    'interval': 'epoch',
                    'name': 'lr'
                } for optimizer in optimizers
            ]
        return optimizers, schedulers
    def training_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, batch_idx,
                                            last_layer=self.model.dec.get_last_layer(), split="train")
        self.log("train/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["train/total_loss"]  
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True,sync_dist=True)#保存日志，由pl_train负责写入日志
        #opt ,_= self.configure_optimizers()
        #opt_ae=self.opt[0]
        if hasattr(self.loss, 'discriminator'):
            opt_ae=self.optimizers(use_pl_optimizer=True)[0]
        else:
            opt_ae=self.optimizers(use_pl_optimizer=True)
        
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()
        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, batch_idx,
                                                last_layer=self.model.dec.get_last_layer(), split="train")
            #opt_disc=self.opt[1]
            opt_disc=self.optimizers(use_pl_optimizer=True)[1]
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()
            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            del log_dict_disc["train/disc_loss"]
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return aeloss
    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, batch_idx,
                                            last_layer=self.model.dec.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        del log_dict_ae["val/total_loss"]
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, batch_idx,
                                                last_layer=self.model.dec.get_last_layer(), split="val")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return self.log_dict
    def on_train_epoch_end(self):
        if hasattr(self.loss, 'discriminator'):
            self.lr_schedulers()[0].step()
            print("当前学习率为：",self.lr_schedulers()[0].get_last_lr())
        else:
            self.lr_schedulers().step()
            print("当前学习率为：",self.lr_schedulers().get_last_lr())
        if hasattr(self.loss, 'discriminator'):
            self.lr_schedulers()[1].step()
            print("当前学习率为：",self.lr_schedulers()[1].get_last_lr())
        #ckpt_path = f"/home/data/haoyi_project/vq_sc/checkpoints/{self.current_epoch:02d}.ckpt"
        #self.trainer.save_checkpoint(ckpt_path)
    def encode_for_experiment(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, id_t, id_b = self.model.encode(x)
        return id_t, id_b
    def decode_for_experiment(self, id_t: torch.Tensor, id_b: torch.Tensor) -> torch.Tensor:
        x_rec = self.model.decode_index(id_t, id_b)
        return x_rec
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        x_rec, _ = self(x)
        log["originals"] = x
        log["reconstructions"] = x_rec
        
        return log