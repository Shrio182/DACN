import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'CIDEr', 'METEOR']

# Dữ liệu chuẩn từ bài báo IEEE TMI 2022 và kết quả thực nghiệm DACN
data = {
    'M2Transformer (Baseline)':        [0.453, 0.290, 0.208, 0.156, 0.365, 0.382, 0.182],
    'R2Gen (EMNLP 2020)':              [0.470, 0.304, 0.219, 0.165, 0.371, 0.398, 0.187],
    'IEEE TMI 2022 (Bài báo gốc)':     [0.496, 0.327, 0.238, 0.178, 0.381, 0.445, 0.203],
    'DACN - IU-Xray Archive (Của bạn)':[0.311, 0.186, 0.123, 0.083, 0.300, 0.327, 0.137],
    'DACN - IU-Xray TimPhoi (Của bạn)':[0.314, 0.205, 0.146, 0.105, 0.289, 0.449, 0.152],
}

colors = ['#95a5a6', '#7f8c8d', '#2ecc71', '#1f77b4', '#ff7f0e']

fig = plt.figure(figsize=(16, 6.5))
fig.suptitle('ĐỐI CHIẾU HIỆU NĂNG THỰC NGHIỆM DACN VỚI BÀI BÁO GỐC (IEEE TMI 2022)',
             fontsize=14, fontweight='bold', y=0.98)

# --- Biểu đồ (A): Grouped Bar Chart ---
ax1 = fig.add_subplot(1, 2, 1)
x = np.arange(len(metrics))
width = 0.16

for i, (label, scores) in enumerate(data.items()):
    offset = (i - 2) * width
    bars = ax1.bar(x + offset, scores, width, label=label, color=colors[i], edgecolor='white', linewidth=0.6)
    if 'Của bạn' in label or 'Bài báo gốc' in label:
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.3f}',
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=6.8, rotation=90, fontweight='bold')

ax1.set_title('(A) So sánh chi tiết từng chỉ số trên tập Test', fontsize=11, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontweight='bold')
ax1.set_ylabel('Điểm số (Score)', fontweight='bold')
ax1.set_ylim(0, 0.58)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=8.5, frameon=True)

# --- Biểu đồ (B): Radar Chart ---
ax2 = fig.add_subplot(1, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

radar_keys = [
    ('IEEE TMI 2022 (Bài báo gốc)', '#2ecc71', '-'),
    ('DACN - IU-Xray Archive (Của bạn)', '#1f77b4', '--'),
    ('DACN - IU-Xray TimPhoi (Của bạn)', '#ff7f0e', '-')
]

for key, col, ls in radar_keys:
    vals = data[key] + data[key][:1]
    ax2.plot(angles, vals, color=col, linewidth=2.2, linestyle=ls, label=key)
    ax2.fill(angles, vals, color=col, alpha=0.12)

ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_thetagrids(np.degrees(angles[:-1]), metrics, fontweight='bold', fontsize=9.5)
ax2.set_ylim(0, 0.55)
ax2.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
ax2.set_yticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'], fontsize=8, color='gray')
ax2.set_title('(B) Biểu đồ phủ đa tiêu chí (Radar Profile)', fontsize=11, fontweight='bold', pad=18)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.94])
os.makedirs('outputs/charts', exist_ok=True)
out_path = 'outputs/charts/paper_comparison_benchmark.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"-> Đã xuất biểu đồ so sánh với bài báo tại: {out_path}")
