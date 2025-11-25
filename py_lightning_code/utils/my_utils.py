import torch
import sionna
import torch
import sys
import torch
import tensorflow as tf
import math
import os
import matplotlib.pyplot as plt
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
def py2tf(tensor: torch.Tensor) -> tf.Tensor:
    """
    将 Python 中的 torch.Tensor 转换为 TensorFlow 中的 tf.Tensor
    """
    dlpack = torch.utils.dlpack.to_dlpack(tensor)
    tf_tensor = tf.experimental.dlpack.from_dlpack(dlpack)
    return tf_tensor
def tf2py(tensor: tf.Tensor) -> torch.Tensor:
    """
    将 TensorFlow 中的 tf.Tensor 转换为 Python 中的 torch.Tensor
    """
    dlpack = tf.experimental.dlpack.to_dlpack(tensor)
    torch_tensor = torch.utils.dlpack.from_dlpack(dlpack)
    return torch_tensor
class split2bitstream:
    """
    用于将pytorch索引张量和比特流张量互相转换
    初始化:
        n: int, 每个数据占用的bit数
        size: tuple, 输入张量的原始形状
        dtype: torch.dtype, 输入张量的dtype
    """
    def __init__(self, n,size,dtype):
        self.len_bit = n
        self.origin_shape= size
        self.dtype=dtype
    def tensor_to_bits(self,x:torch.Tensor) -> torch.Tensor:
        """
        输入[B,....] 输出[B,N*len_bit]
        """
        if x.shape!=self.origin_shape:
            return None
        x=x.view(x.shape[0],-1)
        x_int64=x.to(torch.int64)
        shifts = torch.arange(self.len_bit, dtype=torch.int32, device=x.device)
        # (B,N,1) 右移 (self.len_bit,) 并取最低位 -> (B,N,self.len_bit)
        bits = ((x_int64[..., None] >> shifts) & 1)
        bits=bits.view(bits.shape[0],-1)
        return bits

    def bits_to_tensor(self,bits:torch.Tensor) -> torch.Tensor:
        """
        输入[B,N*len_bit] 输出[B,...]
        """
        bits=bits.to(torch.int64)
        bits=bits.view(-1,bits.shape[1]//self.len_bit,self.len_bit)
        shifts = torch.arange(self.len_bit, dtype=torch.int32, device=bits.device)
        x = (bits.to(torch.int64) * (1 << shifts)).sum(dim=-1)
        x=x.view(self.origin_shape)
        x=x.to(self.dtype)
        return x
   
if __name__ == '__main__':
    x = torch.randint(0,511,(600,),dtype=torch.int64)
    x=x.view(2,2,-1)
    print(x.shape)
    bitstream=split2bitstream(8,x.shape,x.dtype)
    bits=bitstream.tensor_to_bits(x)
    print(bits.shape)
    x=bitstream.bits_to_tensor(bits)
    print(x.shape)
