import torch
import os
import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import pytorch_lightning as pl
import random
import numpy as np
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
from py_lightning_code.utils.general import get_config_from_file, initialize_from_config, setup_callbacks

if __name__ == '__main__':

    # Load configuration
    config = get_config_from_file("/home/data/haoyi_project/vq_sc/config/imagenet_vitvq_base.yaml")

    # Build model
    model = initialize_from_config(config.model)
    data = np.random.randn(1, 3, 256, 256)

# 转成 PyTorch 张量
    data = torch.tensor(data, dtype=torch.float32)
    data= data.to(model.device)
    model(data)