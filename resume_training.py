import torch
import pytorch_lightning as pl
import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from py_lightning_code.utils.general import get_config_from_file, initialize_from_config,resume_callbacks
import torch
from pytorch_lightning.loggers import WandbLogger
print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
"""
继续训练需要五个参数,分别是:
config(确保和之前一致)
checkpoints_path:断点路径
log_path:上一次训练时本地的log所在的文件夹
run_id:wandb日志的唯一标识,索引上一次训练的日志并接着训练
wandb name:可以和之前的保持一致，也可以自己定义
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)#请确保config文件和原来的一致
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ac', '--accelerator', type=str, default="cuda")
    parser.add_argument('-d', '-d', '--devices', type=int, nargs='+', default=[1])
    parser.add_argument('-u', '--update_every', type=int, default=1)
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-b', '--batch_frequency', type=int, default=750)
    parser.add_argument('-m', '--max_images', type=int, default=4)
    parser.add_argument('-cn', '--checkpoint_name', type=str, default="cnn_w_error_0.01_top_500_channel_loss")
    parser.add_argument('-cp', '--checkpoint_path', type=str, default="/home/data/haoyi_project/vq_sc/checkpoints/cnn_w_error_0.01_top_500_channel_loss-epoch=1013.ckpt")
    parser.add_argument('-lp','--log_path', type=str, default="/home/data/haoyi_project/vq_sc/config/control_cnn_w_error_0.01_top_500_channel_loss/24112025_171543")#继续更新日志
    parser.add_argument('-id','--wandb_name', type=str, default="/home/data/haoyi_project/vq_sc/config/control_cnn_w_error_0.01_top_500_channel_loss_24112025_171543")#可以改名
    parser.add_argument('-rd','--run_id', type=str, default="zxc6v68u")#wandb唯一id
    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    config = get_config_from_file(Path("configs")/(args.config+".yaml"))
    exp_config = OmegaConf.create({"name": args.config,  "update_every": args.update_every,
                                    "use_amp": args.use_amp, "batch_frequency": args.batch_frequency,
                                   "max_images": args.max_images,"checkpoint_name": args.checkpoint_name})

    # Build model
    model = initialize_from_config(config.model)
    print(model.learning_rate)
    #model.learning_rate = exp_config.base_lr
    print(model.epochs)
    print(config.model.params.epochs)
    # Setup callbacks
    callbacks, logger = resume_callbacks(exp_config, config,args)
    data = initialize_from_config(config.dataset)
    data.prepare_data()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    trainer = pl.Trainer(
        max_epochs=config.model.params.epochs,
        precision=32,
        callbacks=callbacks,
        accelerator='gpu',
        devices=args.devices,
        strategy="auto",
        accumulate_grad_batches=exp_config.update_every,
        logger=logger,
    )

    # 继续训练
    trainer.fit(model, data,ckpt_path=args.checkpoint_path)
