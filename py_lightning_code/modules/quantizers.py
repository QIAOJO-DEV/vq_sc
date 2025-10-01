# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

import math
from functools import partial
from typing import Tuple, Optional
from py_lightning_code.utils.my_utils import split2bitstream
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseQuantizer(nn.Module):
    def __init__(self, embed_dim: int, n_embed: int, straight_through: bool = True, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None) -> None:
        super().__init__()
        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.normal_()
        
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass
    
    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:#Residual_VQ暂不启用
        if not self.use_residual:
            if not self.with_error:
                z_q, loss, encoding_indices = self.quantize(z)
            else:
                z_q, loss, encoding_indices = self.quantize_with_error(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(partial(torch.stack, dim = -1), (losses, encoding_indices))
            loss = losses.mean()

        # preserve gradients with straight-through estimator
        if self.straight_through:
            z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices


class VectorQuantizer(BaseQuantizer):
    def __init__(self, embed_dim: int, n_embed: int, with_error: bool, error_prob:float,beta: float = 0.25, use_norm: bool = True,
                 use_residual: bool = False, num_quantizers: Optional[int] = None, **kwargs) -> None:
        super().__init__(embed_dim, n_embed, True,
                         use_norm, use_residual, num_quantizers)
        
        self.beta = beta
        self.with_error=with_error
        self.error_prob=error_prob
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)
        #print("没加错误比特")
        return z_qnorm, loss, encoding_indices
    def quantize_with_error(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:#引入比特错误信道来训练
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        encoding_indices = encoding_indices.view(*z.shape[:-1])
        split=split2bitstream(int(math.log2(self.n_embed)),encoding_indices.shape,encoding_indices.dtype)
        encoding_indices=split.tensor_to_bits(encoding_indices)
        flip_mask = torch.rand_like(encoding_indices.float()) < self.error_prob#按照p的概率进行翻转
        encoding_indices = encoding_indices ^ flip_mask.to(encoding_indices.dtype)
        encoding_indices=split.bits_to_tensor(encoding_indices)
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
               torch.mean((z_qnorm - z_norm.detach())**2)
        #print("添加了错误比特")
        return z_qnorm, loss, encoding_indices



class GumbelQuantizer(BaseQuantizer):
    def __init__(self, embed_dim: int, n_embed: int, temp_init: float = 1.0,
                 use_norm: bool = True, use_residual: bool = False, num_quantizers: Optional[int] = None, **kwargs) -> None:
        super().__init__(embed_dim, n_embed, False,
                         use_norm, use_residual, num_quantizers)
        
        self.temperature = temp_init
        
    def quantize(self, z: torch.FloatTensor, temp: Optional[float] = None) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # force hard = True when we are in eval mode, as we must quantize
        hard = not self.training
        temp = self.temperature if temp is None else temp
        
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)

        logits = - torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) - \
                 torch.sum(embedding_norm ** 2, dim=1) + 2 * \
                 torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        logits =  logits.view(*z.shape[:-1], -1)
        
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=-1, hard=hard)
        z_qnorm = torch.matmul(soft_one_hot, embedding_norm)
        
        # kl divergence to the prior loss
        logits =  F.log_softmax(logits, dim=-1) # use log_softmax because it is more numerically stable
        loss = torch.sum(logits.exp() * (logits+math.log(self.n_embed)), dim=-1).mean()
               
        # get encoding via argmax
        encoding_indices = soft_one_hot.argmax(dim=-1)
        
        return z_qnorm, loss, encoding_indices
