import os
import math
import torch
import argparse
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from py_lightning_code.utils.physical_layer import PhysicalLayer
from py_lightning_code.utils.to_patch import split2patch
from py_lightning_code.utils.my_utils import split2bitstream
from py_lightning_code.dataloader.ImageFolder_test import ImageFileDataset
from py_lightning_code.utils.general import get_config_from_file, initialize_from_config
import subprocess
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# =====================
# 评估指标函数
# =====================
def compute_metrics(recon, target, perceptual_loss):
    psnr_val = psnr(target.cpu().numpy(), recon.cpu().numpy(), data_range=1.0)
    ssim_val = ssim(target[0].permute(1,2,0).cpu().numpy(),
                    recon[0].permute(1,2,0).cpu().numpy(),
                    channel_axis=2, data_range=1.0)
    lpips_val = perceptual_loss(recon*2-1, target*2-1).mean().item()
    return psnr_val, ssim_val, lpips_val

# =====================
# 模型加载 + 推理
# =====================
def load_model_from_ckpt(ckpt_path, config_path, device, codebook_path=None):
    config = get_config_from_file(config_path)
    config.model.params.model_param.error_strategy = "none"# 关闭错误注入
    vqvae = initialize_from_config(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    vqvae.load_state_dict(ckpt['state_dict'])
    vqvae.eval()
    if codebook_path is not None:
        new_codebook = torch.load(codebook_path)
        vqvae.model.quantize_b.embedding.weight = torch.nn.Parameter(new_codebook["codebook_b"].to(device))
        vqvae.model.quantize_t.embedding.weight = torch.nn.Parameter(new_codebook["codebook_t"].to(device))
        print(f"Replaced codebook for {ckpt_path}")
    bits_per_index = int(math.log2(config.model.params.model_param.n_embed))
    return vqvae, bits_per_index

# =====================
# 单个模型/方法 SNR 循环推理
# =====================
def evaluate_model(vqvae, dataloader, physical_layer, bits_per_index, SNR_list, device, perceptual_loss):
    results = {"PSNR": [], "SSIM": [], "LPIPS": []}

    for SNR in tqdm(SNR_list, desc="SNR loop"):
        psnr_list, ssim_list, lpips_list = [], [], []

        for batch in dataloader:
            batch_img = batch['image'].to(device)
            with torch.no_grad():
                id_t, id_b = vqvae.encode_for_experiment(batch_img)

                # bits 转换
                split_bit_t = split2bitstream(bits_per_index, id_t.shape, id_t.dtype)
                split_bit_b = split2bitstream(bits_per_index, id_b.shape, id_b.dtype)
                id_t = split_bit_t.tensor_to_bits(id_t)
                id_b = split_bit_b.tensor_to_bits(id_b)

                # patch 分割
                split_patch_t = split2patch(id_t.shape, id_t.dtype)
                split_patch_b = split2patch(id_b.shape, id_b.dtype)
                id_t = split_patch_t.tensor_to_patch(id_t)
                id_b = split_patch_b.tensor_to_patch(id_b)

                # 物理信道模拟
                id_t, _ = physical_layer.pass_channel(id_t, ebno_db=SNR - 10*math.log10(4))
                id_b, _ = physical_layer.pass_channel(id_b, ebno_db=SNR - 10*math.log10(4))

                # patch -> tensor -> bits -> tensor
                id_t = split_patch_t.patch_to_tensor(id_t)
                id_b = split_patch_b.patch_to_tensor(id_b)
                id_t = split_bit_t.bits_to_tensor(id_t)
                id_b = split_bit_b.bits_to_tensor(id_b)

                # 重构
                recon = vqvae.decode_for_experiment(id_t, id_b).clamp(0,1)

                # 计算指标
                ps_val, ss_val, lp_val = compute_metrics(recon, batch_img, perceptual_loss)
                psnr_list.append(ps_val)
                ssim_list.append(ss_val)
                lpips_list.append(lp_val)

        results["PSNR"].append(np.mean(psnr_list))
        results["SSIM"].append(np.mean(ssim_list))
        results["LPIPS"].append(np.mean(lpips_list))

    return results

# =====================
# BPG 方法 SNR 循环推理
# =====================
def encode_bpg(img_path, quality=30, temp_dir="/tmp"):
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
    return torch.tensor(list(bpg_bytes), dtype=torch.uint8), img_tensor.unsqueeze(0)

def decode_bpg_from_tensor(bpg_tensor, temp_dir="/tmp"):
    temp_bpg = os.path.join(temp_dir, "temp_decoded.bpg")
    temp_png = os.path.join(temp_dir, "temp_decoded.png")
    with open(temp_bpg, "wb") as f:
        f.write(bytes(bpg_tensor.tolist()))
    subprocess.run(["bpgdec", temp_bpg, "-o", temp_png], check=True)
    img = Image.open(temp_png).convert("RGB")
    img_np = np.array(img).astype(np.float32)/255.0
    img_tensor = torch.from_numpy(img_np).permute(2,0,1)
    img_tensor = img_tensor.unsqueeze(0)#扩充一个维度
    os.remove(temp_bpg)
    os.remove(temp_png)
    return img_tensor

def evaluate_bpg(dataloader, physical_layer, SNR_list, perceptual_loss, device, quality=30):
    results = {"PSNR": [], "SSIM": [], "LPIPS": []}
    bits_per_index = 8  # BPG 字节 = 8bit

    for SNR in tqdm(SNR_list, desc="BPG SNR loop"):
        psnr_list, ssim_list, lpips_list = [], [], []

        for batch in dataloader:
            batch_img = batch['image'].to(device)
            for i in range(batch_img.shape[0]):
                img_path = batch['path'][i]
                bpg_tensor, img_tensor = encode_bpg(img_path, quality=quality)
                img_tensor = img_tensor.to(device)

                try:
                    # BPG 编码

                    # bits -> patch
                    split_bit = split2bitstream(bits_per_index, bpg_tensor.shape, bpg_tensor.dtype)
                    bits = split_bit.tensor_to_bits(bpg_tensor)
                    split_patch = split2patch(bits.shape, bits.dtype)
                    patch = split_patch.tensor_to_patch(bits)

                    # 信道模拟
                    patch, _ = physical_layer.pass_channel(patch, ebno_db=SNR - 10*math.log10(4))

                    # patch -> bits -> tensor
                    patch = split_patch.patch_to_tensor(patch)
                    bits = split_bit.bits_to_tensor(patch)

                    # BPG 解码
                    recon = decode_bpg_from_tensor(bits.cpu()).to(device)

                except Exception as e:
                    print(f"[Warning] BPG decode failed for {img_path} at SNR={SNR}: {e}")
                    # 用全零图像代替
                    recon = torch.zeros_like(batch_img[i]).unsqueeze(0)#扩充一个维度

                # 计算指标
                ps_val, ss_val, lp_val = compute_metrics(recon, img_tensor, perceptual_loss)
                psnr_list.append(ps_val)
                ssim_list.append(ss_val)
                lpips_list.append(lp_val)

        results["PSNR"].append(np.mean(psnr_list))
        results["SSIM"].append(np.mean(ssim_list))
        results["LPIPS"].append(np.mean(lpips_list))

    return results


# =====================
# 主函数
# =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--SNR_list', type=list, default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_ckpts', type=list, default=['/home/data/haoyi_projects/vq_sc/checkpoints/cnn_wo_error_EMA_GAN_lpips_big-epoch=2932.ckpt','/home/data/haoyi_projects/vq_sc/checkpoints/cnn_wo_error_EMA_GAN_lpips_big-epoch=2932.ckpt'])
    parser.add_argument('--config_files', type=list, default=['/home/data/haoyi_projects/vq_sc/config/control_cnn_wo_error_EMA.yaml','/home/data/haoyi_projects/vq_sc/config/control_cnn_wo_error_EMA.yaml'])
    parser.add_argument('--model_name', type=list, default=['VQ-reassign index','VQ'])
    parser.add_argument('--codebooks', type=list, default=['/home/data/haoyi_projects/vq_sc/reassign_codebook/cnn_wo_error_EMA_GAN_lpips_big-epoch=2932.pt',None])
    parser.add_argument('--pic_dir', type=str, default='/home/data/haoyi_projects/vq_sc/data_set/kodak')
    parser.add_argument('--bpg_quality', type=int, default=30)
    args = parser.parse_args()
    save_dir = "/home/data/haoyi_projects/vq_sc/img_save"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cpu')
    dataset = ImageFileDataset(args.pic_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    physical_layer = PhysicalLayer(num_bits_per_symbol=4)
    perceptual_loss = lpips.LPIPS(net='vgg', verbose=False).to(device)

    # =====================
    # 循环多个 checkpoint
    # =====================
    all_results = {}
    for idx, (ckpt, cfg, model_name) in enumerate(zip(args.model_ckpts, args.config_files, args.model_name)):
        print(f"Processing model {idx}: {ckpt}")
        codebook = args.codebooks[idx] if args.codebooks is not None else None
        vqvae, bits_per_index = load_model_from_ckpt(ckpt, cfg, device, codebook)
        results = evaluate_model(vqvae, dataloader, physical_layer, bits_per_index, args.SNR_list, device, perceptual_loss)
        all_results[f"{model_name}"] = results
    print("Processing BPG baseline...")
    all_results["BPG"] = evaluate_bpg(dataloader, physical_layer, args.SNR_list, perceptual_loss, device, quality=args.bpg_quality)

    metrics = ['PSNR', 'SSIM', 'LPIPS']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for metric in metrics:
        plt.figure(figsize=(10,6))
        for idx, key in enumerate(all_results.keys()):
            plt.plot(args.SNR_list, all_results[key][metric], 'o-', label=key, color=colors[idx])
        plt.xlabel("SNR (dB)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs SNR for multiple checkpoints (with BPG)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_vs_snr.png"))
        plt.show()

