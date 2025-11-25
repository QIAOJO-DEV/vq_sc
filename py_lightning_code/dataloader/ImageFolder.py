
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
"""
自定义的ImageFolder数据集,专门用于处理文件夹下为图片的数据集
"""


class ImageFileDataset(datasets.ImageFolder):
    def __init__(self, root):
        self.transform = T.Compose([
            #T.Resize(256),
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            #T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])#输入范围变为[-1,1]
        ])
        super().__init__(root, transform=self.transform)
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return {'image': sample, 'class': torch.tensor([target])}
