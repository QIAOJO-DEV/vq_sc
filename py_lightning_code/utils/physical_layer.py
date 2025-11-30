import sys
import os
import math
# 将项目根目录加入 sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import torch
from torchvision import transforms
from PIL import Image
import subprocess
import numpy as np
import tensorflow as tf
from py_lightning_code.utils.FEC import Choose_FEC
from py_lightning_code.utils.Modulation import Modulation
from py_lightning_code.utils.to_patch import split2patch
from py_lightning_code.utils.my_utils import split2bitstream
import sionna
from sionna.phy.channel import RayleighBlockFading
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
class PhysicalLayer:
    """
    物理层传输

    """
    def __init__(self, num_bits_per_symbol,modulation_type="qam",fec_type="LDPC",crc_type="CRC24A",channel_type="awgn"):
        k=64
        n=128
        self.fec=Choose_FEC(fec_type,k,n)
        self.num_bits_per_symbol=num_bits_per_symbol
        self.modulation=Modulation(modulation_type,num_bits_per_symbol)
        self.crc_encoder=sionna.phy.fec.crc.CRCEncoder(crc_degree=crc_type)
        self.crc_decoder=sionna.phy.fec.crc.CRCDecoder(crc_encoder=self.crc_encoder)
        if channel_type=="awgn":
            self.channel=sionna.phy.channel.AWGN()
            self.channel_type=channel_type
        elif channel_type=="rayleigh":
            self.channel=sionna.phy.channel.AWGN()
            self.rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)
            self.channel_type=channel_type

    def pass_channel(self,bitstream,ebno_db= 10):
        """
        物理层传输：
        1. CRC编码
        2. FEC编码
        3. 调制
        """
        bitstream=tf.cast(bitstream,tf.float32)
        bitstream=self.crc_encoder(bitstream)
        bitstream, pad_code = self.fec.ldpc_encoder(bitstream)
        modulated_symbols,pad_mod = self.modulation.modulation(bitstream)
        my_channel = self.channel
        if self.channel_type=="awgn":
            no = sionna.phy.utils.ebnodb2no(ebno_db, self.num_bits_per_symbol, coderate=1)
            modulated_symbols=my_channel(modulated_symbols,no)
        elif self.channel_type=="rayleigh":     
            no = sionna.phy.utils.ebnodb2no(ebno_db, self.num_bits_per_symbol, coderate=1)
            a, tau = self.rayleigh(batch_size=modulated_symbols.shape[0], num_time_steps=modulated_symbols.shape[1])  # 生成瑞利衰弱信道
            h = tf.squeeze(a, axis=[1, 2, 3, 4, 5])
            modulated_symbols = modulated_symbols * h  # 应用瑞利衰落信道
            modulated_symbols=my_channel(modulated_symbols,no)
        if self.channel_type=="rayleigh":
            modulated_symbols=modulated_symbols/h
        rec_bitstream=self.modulation.demodulation(modulated_symbols,no,pad_mod)
        rec_bitstream=self.fec.ldpc_decoder(rec_bitstream,pad_code)
        rec_bitstream,valid=self.crc_decoder(rec_bitstream)
        valid = tf.squeeze(valid, axis=-1)
        return rec_bitstream,valid
    
    def harq_transmit(self, bit_patches, ebno_db=1, max_retransmissions=4, mode="TYPE-1"):
        B = tf.shape(bit_patches)[0]
        # 成功掩码
        success_mask = tf.zeros([B], dtype=tf.bool)
        # 接收比特
        rec_bits = tf.zeros_like(bit_patches,dtype=tf.int32)
        accumulated_symbols = None  # TYPE-1/CC/IR
        accumulated_bits = None  # IR可用

        # 1️⃣ CRC编码
        bit_patches = tf.cast(bit_patches, tf.float32)
        bit_patches = self.crc_encoder(bit_patches)
        num_retx = tf.zeros([B], dtype=tf.float32)
        # 2️⃣ FEC编码
        coded_bits, pad_code = self.fec.ldpc_encoder(bit_patches)

        for retx in range(max_retransmissions):
            # 选择需要重传的包
            to_send_mask = tf.logical_not(success_mask)
            num_to_send = tf.reduce_sum(tf.cast(to_send_mask, tf.int32))
            if num_to_send == 0:
                break

            # 挑选要发送的比特
            bits_tx = tf.boolean_mask(coded_bits, to_send_mask)

            # 3️⃣ 调制
            tx_symbols, pad_mod = self.modulation.modulation(bits_tx)

            # 4️⃣ 信道
            no = sionna.phy.utils.ebnodb2no(ebno_db, self.num_bits_per_symbol, coderate=1)
            rx_symbols = tx_symbols

            if self.channel_type == "awgn":
                rx_symbols = self.channel(tx_symbols, no)
            elif self.channel_type == "rayleigh":
                a, tau = self.rayleigh(batch_size=tf.shape(tx_symbols)[0],
                                    num_time_steps=tf.shape(tx_symbols)[1])
                h = tf.squeeze(a, axis=[1, 2, 3, 4, 5])
                rx_symbols = tx_symbols * h
                rx_symbols = self.channel(rx_symbols, no)
                rx_symbols = rx_symbols / h

            # 5️⃣ HARQ 合并
            if mode == "TYPE-1":
                combined_symbols = rx_symbols
            elif mode == "CC":  # Chase Combining
                if accumulated_symbols is None:
                    accumulated_symbols = tf.zeros_like(rx_symbols)
                # 更新累加符号
                indices = tf.where(to_send_mask)
                accumulated_symbols = tf.tensor_scatter_nd_add(accumulated_symbols, indices, rx_symbols)
                # 更新计数器
                num_retx = tf.tensor_scatter_nd_add(num_retx, indices, tf.ones_like(indices[:,0], dtype=tf.float32))
        # 取平均
                combined_symbols = tf.boolean_mask(accumulated_symbols, to_send_mask) / \
                           tf.cast(tf.boolean_mask(num_retx, to_send_mask)[:, None], tf.complex64)  # 广播平均
            elif mode == "IR":  # Incremental Redundancy
                # 每次发送不同编码子集（循环位移）
                redundancy_shift = (retx * 8) % tf.shape(tx_symbols)[1]
                tx_symbols_shifted = tf.roll(tx_symbols, shift=redundancy_shift, axis=1)

                if self.channel_type == "awgn":
                    rx_symbols_shifted = self.channel(tx_symbols_shifted, no)
                elif self.channel_type == "rayleigh":
                    a, tau = self.rayleigh(batch_size=tf.shape(tx_symbols_shifted)[0],
                                        num_time_steps=tf.shape(tx_symbols_shifted)[1])
                    h = tf.squeeze(a, axis=[1, 2, 3, 4, 5])
                    rx_symbols_shifted = tx_symbols_shifted * h
                    rx_symbols_shifted = self.channel(rx_symbols_shifted, no)
                    rx_symbols_shifted = rx_symbols_shifted / h

                if accumulated_symbols is None:
                    accumulated_symbols = tf.zeros_like(rx_symbols_shifted)
                indices = tf.where(to_send_mask)
                accumulated_symbols = tf.tensor_scatter_nd_add(accumulated_symbols, indices, rx_symbols_shifted)
                combined_symbols = tf.boolean_mask(accumulated_symbols, to_send_mask)
            else:
                raise ValueError("Unsupported HARQ mode")

            # 6️⃣ 解调
            rec_bits_now = self.modulation.demodulation(combined_symbols, no, pad_mod)

            # 7️⃣ FEC 解码
            rec_bits_now = self.fec.ldpc_decoder(rec_bits_now, pad_code)

            # 8️⃣ CRC 检查
            rec_bits_now, valid = self.crc_decoder(rec_bits_now)
            rec_bits_now = tf.cast(rec_bits_now, tf.int32)

            # 更新接收结果
            rec_bits = tf.tensor_scatter_nd_update(rec_bits, tf.where(to_send_mask), rec_bits_now)
            valid = tf.squeeze(valid, axis=-1)
            success_mask = tf.tensor_scatter_nd_update(success_mask, tf.where(to_send_mask), valid)

        return rec_bits, success_mask
TEMP_DIR='/home/data/haoyi_projects/vq_sc'
def encode_bpg(img_path, quality=30, temp_dir=TEMP_DIR):
    img = Image.open(img_path).convert("RGB")
    crop_size = 256  
    # 中心裁剪
    W, H = img.size
    left = (W - crop_size) // 2
    top = (H - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    img_np = np.array(img).astype(np.float32) / 255.0   # HWC
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # CHW
    # 保存临时 PNG
    temp_png = os.path.join(temp_dir, "temp_crop.png")
    img.save(temp_png)
    bpg_path = os.path.join(temp_dir, os.path.basename(img_path).split('.')[0] + ".bpg")
    subprocess.run(["bpgenc", temp_png, "-o", bpg_path, "-q", str(quality)], check=True)
    with open(bpg_path, "rb") as f:
        bpg_bytes = f.read()
    os.remove(bpg_path)
    os.remove(temp_png)
    return torch.tensor(list(bpg_bytes), dtype=torch.uint8),img_tensor

def decode_bpg_from_tensor(bpg_tensor, temp_dir=TEMP_DIR):
    temp_bpg = os.path.join(temp_dir, "temp_decoded.bpg")
    temp_png = os.path.join(temp_dir, "temp_decoded.png")
    with open(temp_bpg, "wb") as f:
        f.write(bytes(bpg_tensor.tolist()))
    subprocess.run(["bpgdec", temp_bpg, "-o", temp_png], check=True)
    img = Image.open(temp_png).convert("RGB")
    img_np = np.array(img).astype(np.float32)/255.0
    img_tensor = torch.from_numpy(img_np).permute(2,0,1)
    img_tensor = img_tensor.unsqueeze(0)#扩充一个维度
    #os.remove(temp_bpg)
    #os.remove(temp_png)
    return img_tensor
def compute_metrics(recon, target, perceptual_loss):
    psnr_val = psnr(target.cpu().numpy(), recon.cpu().numpy(), data_range=1.0)
    ssim_val = ssim(target[0].permute(1,2,0).cpu().numpy(),
                    recon[0].permute(1,2,0).cpu().numpy(),
                    channel_axis=2, data_range=1.0)
    lpips_val = perceptual_loss(recon*2-1, target*2-1).mean().item()
    return psnr_val, ssim_val, lpips_val


if __name__ == "__main__":
    #x = torch.randint(0,2,(33,1),dtype=torch.int64)
    img_path='/home/data/haoyi_projects/vq_sc/data_set/kodak/kodak_test/kodim01.png'
    SNR=8
    x,target_img=encode_bpg(img_path)
    target_img=target_img.unsqueeze(0)
    print(x)
    split_bit=split2bitstream(8,x.shape,x.dtype)
    x=split_bit.tensor_to_bits(x)
    split_patch=split2patch(x.shape,x.dtype)
    x=split_patch.tensor_to_patch(x)
    physical_layer=PhysicalLayer(num_bits_per_symbol=4,channel_type="awgn")
    #rec_bitstream,valid=physical_layer.pass_channel(x,ebno_db=1)
    rec_bitstream,valid=physical_layer.harq_transmit(x,mode="TYPE-1",ebno_db=SNR - 10*math.log10(4))
    print(valid)
    rec_bitstream=split_patch.patch_to_tensor(rec_bitstream)
    rec_bitstream=split_bit.bits_to_tensor(rec_bitstream)
    decoded_img=decode_bpg_from_tensor(rec_bitstream)
    perceptual_loss = lpips.LPIPS(net='vgg', verbose=False)
    print(decoded_img.shape,target_img.shape)
    psnr_val, ssim_val, lpips_val = compute_metrics(decoded_img, target_img, perceptual_loss)
    print(f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, LPIPS: {lpips_val:.4f}")

    #print(rec_bitstream)
    #print(valid)
