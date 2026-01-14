import os
import math
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lpips

from py_lightning_code.utils.physical_layer import PhysicalLayer
from py_lightning_code.utils.to_patch import split2patch
from py_lightning_code.utils.my_utils import split2bitstream
from py_lightning_code.utils.general import get_config_from_file, initialize_from_config
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# =====================
# 指标计算函数 (基于你的要求)
# =====================
def compute_metrics(recon, target, perceptual_loss):
    """
    recon, target: Tensor (B, C, H, W), range [0, 1]
    """
    # 计算 PSNR (skimage 接受 numpy)
    psnr_val = psnr(target.cpu().numpy(), recon.cpu().numpy(), data_range=1.0)
    
    # 计算 SSIM
    ssim_vals = []
    for i in range(target.shape[0]):
        ssim_vals.append(
            ssim(
                target[i].permute(1, 2, 0).cpu().numpy(),
                recon[i].permute(1, 2, 0).cpu().numpy(),
                channel_axis=2,
                data_range=1.0
            )
        )
    ssim_val = sum(ssim_vals) / len(ssim_vals)
    
    # 计算 LPIPS (需要输入在 [-1, 1] 范围)
    lpips_val = perceptual_loss(recon * 2 - 1, target * 2 - 1).mean().item()
    
    return psnr_val, ssim_val, lpips_val

# =====================
# 工具函数
# =====================
def load_image(img_path, device, crop_size=2048):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    left = (W - crop_size) // 2
    top = (H - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    img = img.crop((left, top, right, bottom))

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_tensor

def load_model_from_ckpt(ckpt_path, config_path, device, codebook_path=None):
    config = get_config_from_file(config_path)
    config.model.params.model_param.error_strategy = "none"
    vqvae = initialize_from_config(config.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    vqvae.load_state_dict(ckpt['state_dict'])
    vqvae.eval()
    if codebook_path is not None:
        new_codebook = torch.load(codebook_path)
        vqvae.model.quantize_b.embedding.weight = torch.nn.Parameter(new_codebook["codebook_b"].to(device))
        vqvae.model.quantize_t.embedding.weight = torch.nn.Parameter(new_codebook["codebook_t"].to(device))
    bits_per_index = int(math.log2(config.model.params.model_param.n_embed))
    return vqvae, bits_per_index

def run_single(vqvae, img, physical_layer, bits_per_index, SNR, device, harq_type):
    with torch.no_grad():
        id_t, id_b = vqvae.encode_for_experiment(img)

        split_bit_t = split2bitstream(bits_per_index, id_t.shape, id_t.dtype)
        split_bit_b = split2bitstream(bits_per_index, id_b.shape, id_b.dtype)
        id_t = split_bit_t.tensor_to_bits(id_t).to(device)
        id_b = split_bit_b.tensor_to_bits(id_b).to(device)

        split_patch_t = split2patch(id_t.shape, id_t.dtype)
        split_patch_b = split2patch(id_b.shape, id_b.dtype)
        id_t = split_patch_t.tensor_to_patch(id_t)
        id_b = split_patch_b.tensor_to_patch(id_b)

        ebno = SNR - 10 * math.log10(4)
        if harq_type == 'none':
            id_t, _ = physical_layer.pass_channel(id_t, ebno_db=ebno)
            id_b, _ = physical_layer.pass_channel(id_b, ebno_db=ebno)
        else:
            id_t, _ = physical_layer.harq_transmit(id_t, mode=harq_type, ebno_db=ebno)
            id_b, _ = physical_layer.harq_transmit(id_b, mode=harq_type, ebno_db=ebno)

        id_t = split_patch_t.patch_to_tensor(id_t)
        id_b = split_patch_b.patch_to_tensor(id_b)
        id_t = split_bit_t.bits_to_tensor(id_t)
        id_b = split_bit_b.bits_to_tensor(id_b)

        recon = vqvae.decode_for_experiment(id_t, id_b).clamp(0, 1)

    return recon

# =====================
# 主函数
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='/home/data/haoyi_projects/vq_sc/data_set/CMSB_train/CMSB_png/DJI_20250429113018_0010_V.png')
    parser.add_argument('--SNR_list', nargs='+', type=float, default=[2,5,8,11,14])
    parser.add_argument('--channel_type', type=str, default='rayleigh')
    parser.add_argument('--harq_type', type=str, default='none')
    parser.add_argument('--model_ckpts', type=list, default=['/home/data/haoyi_projects/vq_sc/checkpoints/low_space_wo_error-epoch=1497.ckpt','/home/data/haoyi_projects/vq_sc/checkpoints/low_space_top_500_0.01_channel_loss-epoch=1454.ckpt'])
    parser.add_argument('--config_files', type=list, default=['./config/low_space_wo_error.yaml','./config/low_space_top_500_0.01_channel_loss.yaml'])
    parser.add_argument('--model_names', type=list, default=['VQ-DeepSC','RDV-SC'])
    parser.add_argument('--codebooks', type=list, default=[None,'/home/data/haoyi_projects/vq_sc/reassign_codebook/low_space_top_500_0.01_channel_loss-epoch=1454.pt'])

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 LPIPS 模型
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    img = load_image(args.img_path, device)
    physical_layer = PhysicalLayer(num_bits_per_symbol=4, channel_type=args.channel_type)

    models = []
    for idx, (ckpt, cfg, name) in enumerate(zip(args.model_ckpts, args.config_files, args.model_names)):
        codebook = args.codebooks[idx] if args.codebooks is not None else None
        model, bpi = load_model_from_ckpt(ckpt, cfg, device, codebook)
        models.append((name, model, bpi))   

    # 设置画布大小（增加高度以容纳多行文字指标）
    fig, axes = plt.subplots(len(models), len(args.SNR_list)+1, figsize=(3.5*(len(args.SNR_list)+1), 4.5*len(models)))

    for r, (name, model, bpi) in enumerate(models):
        # 原始图像列
        axes[r,0].imshow(img[0].permute(1,2,0).cpu())
        axes[r,0].set_title(f"{name}\nGround Truth", fontsize=18)
        axes[r,0].axis('off')

        for c, snr in enumerate(args.SNR_list):
            # 推理获得重构图
            recon = run_single(model, img, physical_layer, bpi, snr, device, args.harq_type)
            
            # 计算指标
            psnr_v, ssim_v, lpips_v = compute_metrics(recon, img, perceptual_loss_fn)
            
            # 显示图像
            axes[r,c+1].imshow(recon[0].permute(1,2,0).cpu())
            
            # 在标题中显示 SNR 和计算出的指标
            title_text = (f"SNR={snr}dB\n"
                          f"SSIM: {ssim_v:.4f}\n"
                          f"LPIPS: {lpips_v:.4f}")
            axes[r,c+1].set_title(title_text, fontsize=18)
            axes[r,c+1].axis('off')

    plt.tight_layout()
    save_path = "./snr_visualization_with_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] Visualization with metrics saved to: {save_path}")

if __name__ == "__main__":
    main()