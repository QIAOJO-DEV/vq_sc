import math
import torch
import tensorflow as tf
import numpy as np
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
class split2patch:
    """
    用于将张量的形状转换为(B,PATACH),并进行padding,对于不同形状的输入要建立不同的对象来变为patch和解patch,同时从py变tf
    """
    def __init__(self,size,dtype,patch_size=1024):
        self.patch_size = int(patch_size)
        self.origin_shape=size
        self.total=np.prod(size)
        self.patch_num=math.ceil(self.total/self.patch_size)
        self.padded_len=self.patch_num*self.patch_size
        self.pad_length=self.padded_len-self.total  
        self.dtype=dtype
    def tensor_to_patch(self, tensor):
        # 拉平成 1D
        tensor=py2tf(tensor)
        tensor = tf.reshape(tensor, [-1])
        # 如果需要，补 0
        if self.pad_length != 0:
            pad = tf.zeros([self.pad_length], dtype=tensor.dtype)
            tensor = tf.concat([tensor, pad], axis=0)
        # 分块
        tensor = tf.reshape(tensor, [-1, self.patch_size])
        return tensor

    def patch_to_tensor(self, tensor):
        # 拉平成 1D
        tensor = tf.reshape(tensor, [-1])
        # 裁掉 padding
        tensor = tensor[:self.total]
        # 恢复原始形状
        tensor = tf.reshape(tensor, self.origin_shape)
        tensor = tf2py(tensor)
        tensor= tensor.to(self.dtype)
        return tensor
if __name__ == '__main__':
    x = torch.randint(0,2,(33,1),dtype=torch.int64)
    patch=split2patch(x.shape,x.dtype)
    x=patch.tensor_to_patch(x)
    print(x.shape)
    x=patch.patch_to_tensor(x)
    print(x.shape)
