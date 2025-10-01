from typing import List, Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

import PIL
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms as T
import pytorch_lightning as pl

from .layers import ViTEncoder as Encoder, ViTDecoder as Decoder
from .quantizers import VectorQuantizer, GumbelQuantizer
from py_lightning_code.utils.general import initialize_from_config


class ViTVQ(pl.LightningModule):
    def __init__(self, image_key: str, image_size: int, patch_size: int, encoder: OmegaConf, decoder: OmegaConf, quantizer: OmegaConf,
                 loss: OmegaConf, learning_rate:float=1e-4,epochs:int=500,path: Optional[str] = None) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.path = path
        self.image_key = image_key
        self.automatic_optimization = False#开启手动优化
        self.loss = initialize_from_config(loss)
        self.encoder = Encoder(image_size=image_size, patch_size=patch_size, **encoder)
        self.decoder = Decoder(image_size=image_size, patch_size=patch_size, **decoder)
        self.quantizer = VectorQuantizer(**quantizer)
        self.pre_quant = nn.Linear(encoder.dim, quantizer.embed_dim)
        self.post_quant = nn.Linear(quantizer.embed_dim, decoder.dim)
        #self.opt ,self.lr_scheduler = self.configure_optimizers()
        if path is not None:
            self.init_from_ckpt(path)
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:    
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        return dec, diff
    def init_from_ckpt(self, path: str):#还是考虑使用check_point
        sd = torch.load(path)["state_dict"]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")#先不考虑部分载入
    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)
        
        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)
        
        return dec
    def get_input(self, batch: Tuple[Any, Any], key: str = 'image') -> Any:
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()

        return x.contiguous()
    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate
        optim_groups = list(self.encoder.parameters()) + \
                       list(self.decoder.parameters()) + \
                       list(self.pre_quant.parameters()) + \
                       list(self.post_quant.parameters()) + \
                       list(self.quantizer.parameters())
        
        optimizers = [torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        schedulers = []
        warmup_epoches =5
        if hasattr(self.loss, 'discriminator'):
            optimizers.append(torch.optim.AdamW(self.loss.discriminator.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4))
            
        schedulers = [
                {
                    'scheduler': lr_scheduler.SequentialLR(optimizer, schedulers=[
                        lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epoches),
                        lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs-warmup_epoches,eta_min=1e-6)
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
                                            last_layer=self.decoder.get_last_layer(), split="train")
        self.log("train/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        del log_dict_ae["train/total_loss"]  
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)#保存日志，由pl_train负责写入日志
        #opt ,_= self.configure_optimizers()
        #opt_ae=self.opt[0]
        opt_ae=self.optimizers(use_pl_optimizer=True)[0]
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()
        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, batch_idx,
                                                last_layer=self.decoder.get_last_layer(), split="train")
            #opt_disc=self.opt[1]
            opt_disc=self.optimizers(use_pl_optimizer=True)[1]
            opt_disc.zero_grad()
            self.manual_backward(discloss)
            opt_disc.step()
            self.log("train/disc_loss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            del log_dict_disc["train/disc_loss"]
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss
    def validation_step(self, batch: Tuple[Any, Any], batch_idx: int) -> torch.FloatTensor:
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step, batch_idx,
                                            last_layer=self.decoder.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/total_loss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        del log_dict_ae["val/rec_loss"]
        del log_dict_ae["val/total_loss"]
        if hasattr(self.loss, 'discriminator'):
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step, batch_idx,
                                                last_layer=self.decoder.get_last_layer(), split="val")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return self.log_dict
    def on_train_epoch_end(self):
        self.lr_schedulers()[0].step()
        print("当前学习率为：",self.lr_schedulers()[0].get_last_lr())
        if hasattr(self.loss, 'discriminator'):
            self.lr_schedulers()[1].step()
            print("当前学习率为：",self.lr_schedulers()[1].get_last_lr())
        #ckpt_path = f"/home/data/haoyi_project/vq_sc/checkpoints/{self.current_epoch:02d}.ckpt"
        #self.trainer.save_checkpoint(ckpt_path)
    def log_images(self, batch: Tuple[Any, Any], *args, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device)
        quant, _ = self.encode(x)
        
        log["originals"] = x
        log["reconstructions"] = self.decode(quant)
        
        return log