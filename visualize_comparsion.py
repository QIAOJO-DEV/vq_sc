import matplotlib.pyplot as plt
import pickle
import os
all_results_path='/home/data/haoyi_projects/vq_sc/experiment_data/RDV-SC_rayleigh_index_assignment_kodak.pkl'
with open(all_results_path, 'rb') as f:
    all_results = pickle.load(f)
deep_jscc_path='/home/data/haoyi_projects/vq_sc/deep_jscc_data/awgn_snr_1_15.pkl'
with open(deep_jscc_path, 'rb') as f:
    a= pickle.load(f)
print(a.keys())
save_dir='./'
marker_list = ['s', 's','D','D','v','v']
all_keys = list(all_results.keys())
marker_map = {key: marker_list[i % len(marker_list)] for i, key in enumerate(all_keys)}
b=False
deep_jscc_results=a["MyModel"]
SNR_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
print(deep_jscc_results)
metrics = ['PSNR', 'SSIM', 'LPIPS']
for metric in metrics:
    plt.figure(figsize=(10,8))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    for key, value in all_results.items():
        plt.plot(
                SNR_list,
                value['results'][metric],
                label=key,
                color=value['color'],
                linestyle=value['linestyle'],
                marker=marker_map[key],
                linewidth=2
        )
    if b:
        plt.plot(
            SNR_list,
            deep_jscc_results['results'][metric],
            marker='o',  # DeepJSCC 可以用不同 marker，便于区分
            linestyle=deep_jscc_results.get('linestyle', '-'),
            color=deep_jscc_results.get('color', 'black'),
            label='DeepJSCC',
            linewidth=2
        )

    plt.xlabel("SNR (dB)", fontweight='bold', fontsize=20)
    plt.ylabel(metric, fontweight='bold', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"{metric} vs SNR(Rayleigh channel)", fontweight='bold', fontsize=20)
    plt.grid(True)
    plt.legend(prop={'size':15, 'weight':'bold'})
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric}_vs_snr.png"))
    plt.show()
fig, axes = plt.subplots(1, 3, figsize=(24, 7)) 

# 将 metrics 和对应的子图轴绑定在一起
for ax, metric in zip(axes, metrics):
    
    # 设置子图边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # 绘制 all_results
    for key, value in all_results.items():
        ax.plot(
            SNR_list,
            value['results'][metric],
            label=key,
            color=value['color'],
            linestyle=value['linestyle'],
            marker=marker_map[key],
            linewidth=2
        )
    
    # 绘制 DeepJSCC (如果 b 为 True)
    if b:
        ax.plot(
            SNR_list,
            deep_jscc_results['results'][metric],
            marker='o',
            linestyle=deep_jscc_results.get('linestyle', '-'),
            color=deep_jscc_results.get('color', 'black'),
            label='DeepJSCC',
            linewidth=2
        )

    # 针对每个子图 (ax) 进行设置
    ax.set_xlabel("SNR (dB)", fontweight='bold', fontsize=20)
    ax.set_ylabel(metric, fontweight='bold', fontsize=20)
    ax.tick_params(axis='both', labelsize=18) # 统一设置刻度大小
    ax.set_title(f"{metric} vs SNR (AWGN)", fontweight='bold', fontsize=18)
    ax.grid(True)
    
    # 也可以选择只在最后一张图画图例，或者每张都画
    ax.legend(prop={'size': 12, 'weight': 'bold'})

# 2. 自动调整间距，防止子图重叠
plt.tight_layout()

# 3. 保存整张大图
save_path = os.path.join(save_dir, "combined_metrics_vs_snr.png")
plt.savefig(save_path, dpi=300) # dpi=300 保证清晰度
plt.show()