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
            T.RandomCrop(256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        super().__init__(root, transform=self.transform)

    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        img_path, _ = self.samples[index]  # self.samples里存的是 (path, class)
        return {'image': sample, 'class': torch.tensor([target]), 'path': img_path}
