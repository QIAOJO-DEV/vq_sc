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
        self.norm = lambda x: F.normalize(x, dim=-1,eps=1e-7) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers

        self.embed_dim = embed_dim
        self.n_embed = n_embed

        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)#512,64
        self.embedding.weight.data.normal_()
        
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass
    def quantize_with_error(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # 重塑输入并归一化
        pass
    def quantize_to_top_k(self, z: torch.FloatTensor, top_k: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        pass
    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:#Residual_VQ暂不启用
        if not self.use_residual:
            if self.error_strategy=="top_k":
                z_q, loss, encoding_indices = self.quantize_to_top_k(z,self.top_k)
            elif self.error_strategy=="bit_flip":
                z_q, loss, encoding_indices = self.quantize_with_error(z)
            elif self.error_strategy=="none":
                z_q, loss, encoding_indices = self.quantize(z)
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
    def __init__(self, embed_dim: int, n_embed: int, error_strategy: str='top_k', error_prob:float=0.05,top_k:int=500,beta: float = 0.25, use_norm: bool = False,
                 use_residual: bool = False, num_quantizers: Optional[int] = None,channel_loss_weight:float=0.01,**kwargs) -> None:
        super().__init__(embed_dim, n_embed, True,
                         use_norm, use_residual, num_quantizers)
        n_bits=int(math.log2(self.n_embed))
        indices = torch.arange(self.n_embed, dtype=torch.int32).unsqueeze(1)  # (n_embed,1)
        bits = ((indices >> torch.arange(n_bits)) & 1).to(torch.uint8)   # (n_embed,n_bit)
        bits_i = bits.unsqueeze(1)  # (n_embed,1,n_bit)
        bits_j = bits.unsqueeze(0) # (1,n_embed,n_bit)
        self.dist_matrix = (bits_i ^ bits_j).sum(dim=-1)#用于加速计算channel_loss的距离矩阵
        self.channel_loss_weight=channel_loss_weight
        self.beta = beta
        self.error_strategy=error_strategy
        self.error_prob=error_prob
        self.top_k=top_k
    def calculate_codebook_loss_vectorized(self,codebook, z, p):
        """
        计算码本损失l_ch
        """
        z=z.detach()#防止z的梯度回传，这一项只优化码本分布
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
        torch.sum(codebook ** 2, dim=1) - 2 * \
        torch.einsum('bd,nd->bn', z, codebook)
    
        B, N = d.shape
        n_bits=int(math.log2(N))
        with torch.no_grad():
            encoding_indices = torch.argmin(d, dim=1).to(codebook.device)  # (B,)
            dist_matrix=self.dist_matrix.to(codebook.device).detach() 
            H = dist_matrix[encoding_indices].to(codebook.device)  # (B,N) 计算出当前索引到其他索引的汉明距离，直接取出汉明矩阵中的某一行
            p = torch.tensor(p, device=H.device, dtype=torch.float32)
    
            logW = H * torch.log(p) + (n_bits - H) * torch.log(1 - p)#防止N较大，p较小时导致的数值不稳定,计算log L
            W = torch.exp(logW)                                  # (B, N)
            W = W.masked_fill(H == 0, 0).to(codebook.device)                       # 排除自身
        distance_loss = torch.sum(W * d, dim=1)   # (B,)
    
        loss = distance_loss.mean()
        return loss
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.contiguous().view(-1, self.embed_dim))
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
        return z_q, loss, encoding_indices#返回的是z_q而不是归一化后的z_qnorm
    def quantize_with_error(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        #z=torch.clamp(z, -10.0, 10.0)
        #z=torch.tanh(z) * 15
        #z = z / (1 + 0.05* z.abs())
        z_reshaped_norm = self.norm(z.contiguous().view(-1, self.embed_dim))
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
        loss+=self.channel_loss_weight*self.calculate_codebook_loss_vectorized(self.embedding.weight, z_reshaped_norm.detach(), self.error_prob)
        #print("添加了错误比特")
        if loss>1000:
            print("encoding_indices min/max:", encoding_indices.min(), encoding_indices.max())
            print("z_q min/max:", z_q.min(), z_q.max())
            print("z min/max:", z.min(), z.max())
            print("z_qnorm min/max:", z_qnorm.min(), z_qnorm.max())
            print("z norm min/max:", z_norm.min(), z_norm.max())
        return z_q, loss, encoding_indices
    def quantize_to_top_k(self,z:torch.FloatTensor,k:int=1)->Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:#随机跳到K个最邻近索引中的某一个，稳定训练
        #z=torch.clamp(z, -10.0, 10.0)
        #z=torch.tanh(z) * 15
        #z = z / (1 + 0.05* z.abs())
        z_reshaped_norm = self.norm(z.contiguous().view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        topk_dist, topk_idx = torch.topk(d, k, dim=1, largest=False)
        B = topk_idx.shape[0]
        p = torch.rand(B, device=z.device)
        error_mask = (p < self.error_prob)
        rand_idx = torch.randint(1, k, (B,), device=z.device) # 随机选择一个索引
        selected_idx = topk_idx[:, 0]
        selected_idx[error_mask] = topk_idx[error_mask, rand_idx[error_mask]]
        encoding_indices = selected_idx.view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm)**2) +  \
                torch.mean((z_qnorm - z_norm.detach())**2)
        loss+=self.channel_loss_weight*self.calculate_codebook_loss_vectorized(self.embedding.weight, z_reshaped_norm.detach(), self.error_prob)
        if loss>200 or loss==float('nan'):
            print("encoding_indices min/max:", encoding_indices.min(), encoding_indices.max())
            print("z_q min/max:", z_q.min(), z_q.max())
            print("z min/max:", z.min(), z.max())
            print("z_qnorm min/max:", z_qnorm.min(), z_qnorm.max())
            print("z norm min/max:", z_norm.min(), z_norm.max())
        return z_q, loss, encoding_indices#返回的是z_q而不是归一化后的z_qnorm
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embedding.weight)

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
class EMAQuantizer(BaseQuantizer):
    def __init__(self, embed_dim: int, n_embed: int, error_strategy: str='none', error_prob:float=0.05,top_k:int=500,beta: float = 0.25, use_norm: bool = False,
                 use_residual: bool = False, num_quantizers: Optional[int] = None, decay=0.99,eps=1e-5,**kwargs) -> None:
        super().__init__(embed_dim, n_embed, True,  # 使用straight_through
                         use_norm, use_residual, num_quantizers)
        
        self.beta = beta
        self.error_strategy=error_strategy
        self.error_prob = error_prob
        self.decay = decay
        self.eps = eps
        # 初始化EMA相关的缓冲区
        # embedding_size[n_embed,embed_dim]
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.embedding.weight.data.clone())
        
    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        # 重塑输入并归一化
        z_reshaped = z.contiguous().view(-1, self.embed_dim)
        z_reshaped_norm = self.norm(z_reshaped)
        
        embedding_weights = self.embedding.weight
        
        embedding_norm = self.norm(embedding_weights)
        
        # 计算距离并获取最近的码本索引
        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        
        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices_reshaped = encoding_indices.view(*z.shape[:-1])
        
        # 获取量化后的向量
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = self.norm(z_q), self.norm(z)
        
        # 在训练模式下，更新EMA
        if self.training:
            # 计算损失（与VectorQuantizer相同的损失函数）
            loss = torch.mean((z_qnorm.detach() - z_norm)**2)
            
            # 计算EMA更新
            with torch.no_grad():
                # 创建one-hot编码
                encodings = torch.zeros(encoding_indices.shape[0], self.n_embed, device=z.device)
                encodings.scatter_(1, encoding_indices, 1)
                
                # 更新
                self.cluster_size.data.mul_(self.decay).add_(torch.sum(encodings, dim=0), alpha=1 - self.decay)
                
                # 码本向量的前进方向
                dw = torch.matmul(encodings.t(), z_reshaped)
                
                # 更新嵌入向量的平均值
                self.embed_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                # 归一化嵌入向量
                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embedding.weight.data.copy_(embed_normalized)
        else:
            # 推理时不计算损失
            loss = torch.tensor(0.0, device=z.device)
        
        return z_qnorm, loss, encoding_indices_reshaped
        
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embedding.weight)
    
