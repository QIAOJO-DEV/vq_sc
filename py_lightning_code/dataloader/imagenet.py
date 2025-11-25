# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import PIL
from typing import Any, Tuple, Union, Optional, Callable

import torch
from torchvision import transforms as T
from torchvision.datasets import ImageNet
from datasets import load_from_disk
from datasets import load_dataset

"""class ImageNetBase(ImageNet):
    def __init__(self, root: str, split: str,
                 transform: Optional[Callable] = None) -> None:
        super().__init__(root=root, split=split, transform=transform)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = super().__getitem__(index)

        return {'image': sample, 'class': torch.tensor([target])}


class ImageNetTrain(ImageNetBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75) -> None:

        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        
        super().__init__(root=root, split='train', transform=transform)
        
"""
class ImageNetTrain():
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256) -> None:
        self.root=root
        self.ds=load_from_disk(root)['train']
        self.resolution=resolution
        self.transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample=self.ds[index]['image']
        target=self.ds[index]['label']
        if self.transform:
            sample=self.transform(sample)
        return {'image': sample, 'class': torch.tensor([target])}

'''class ImageNetValidation(ImageNetBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,) -> None:

        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        
        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor()
        ])
        
        super().__init__(root=root, split='val', transform=transform)
'''
class ImageNetValidation():
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,) -> None:
        self.root=root
        self.ds=load_from_disk(root)['validation']
        self.resolution=resolution
        self.transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample=self.ds[index]['image']
        target=self.ds[index]['label']
        if self.transform:
            sample=self.transform(sample)
        return {'image': sample, 'class': torch.tensor([target])}
