# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
import torch
import pytorch_lightning as pl
import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from py_lightning_code.utils.general import get_config_from_file, initialize_from_config, setup_callbacks
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-nn', '--num_nodes', type=int, default=1)
    parser.add_argument('-ac', '--accelerator', type=str, default="cuda")
    parser.add_argument('-d', '-d', '--devices', type=int, nargs='+', default=[0,1])
    parser.add_argument('-u', '--update_every', type=int, default=1)
    #parser.add_argument('-e', '--epochs', type=int, default=500)epochs在配置文件中设置
    #parser.add_argument('-lr', '--base_lr', type=float, default=4.5e-6)学习率在配置文件中设置
    parser.add_argument('-a', '--use_amp', default=False, action='store_true')
    parser.add_argument('-b', '--batch_frequency', type=int, default=750)
    parser.add_argument('-m', '--max_images', type=int, default=4)
    parser.add_argument('-cn', '--checkpoint_name', type=str, default="model_checkpointsc")
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
    callbacks, logger = setup_callbacks(exp_config, config)

    # Build data modules
    data = initialize_from_config(config.dataset)
    data.prepare_data()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # Build trainer
    trainer = pl.Trainer(max_epochs=config.model.params.epochs,
                         precision=32,
                         callbacks=callbacks,
                         accelerator='gpu',
                         devices=args.devices,
                         strategy="ddp_find_unused_parameters_true",
                         accumulate_grad_batches=exp_config.update_every,
                         logger=logger)

    # Train
    trainer.fit(model, data)
