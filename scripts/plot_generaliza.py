import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 准备数据 (从你的实验结果中提取)
# ==========================================
# 指标名称
metrics = ['AUC', 'Specificity']

# Transformer 在 Kaggle 上的表现 (Mean, Std)
transformer_means = [0.624, 0.656]
transformer_stds = [0.033, 0.279]

# Random Forest 在 Kaggle 上的表现 (Mean, Std) - 根据你刚刚跑出的最新结果
rf_means = [0.922, 0.952]
rf_stds = [0.004, 0.008]

# ==========================================
# 2. 全局绘图风格设置 (学术标准)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# ==========================================
# 3. 绘制分组柱状图
# ==========================================
x = np.arange(len(metrics))  # 指标的 x 轴位置
width = 0.35  # 柱子的宽度

# 绘制 Transformer 柱子 (冷色调：蓝色)
rects1 = ax.bar(x - width/2, transformer_means, width,
                yerr=transformer_stds, capsize=5,
                label='Transformer (Agent-NAS)',
                color='#5b9bd5', edgecolor='black', zorder=3)

# 绘制 Random Forest 柱子 (暖色调：橙红)
rects2 = ax.bar(x + width/2, rf_means, width,
                yerr=rf_stds, capsize=5,
                label='Random Forest (Baseline)',
                color='#ed7d31', edgecolor='black', zorder=3)

# ==========================================
# 4. 图表修饰与标注
# ==========================================
# 设置 Y 轴范围和标签
ax.set_ylim(0, 1.25)
ax.set_ylabel('Score', fontweight='bold', fontsize=16)
ax.set_title('External Validation on Kaggle Cohort (n=918)\nZero-padded Artifacts',
             fontweight='bold', fontsize=16, pad=15)

# 设置 X 轴刻度
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontweight='bold', fontsize=16)

# 添加图例
ax.legend(loc='upper left', fontsize=12, framealpha=0.9, edgecolor='gray')

# 添加背景网格线 (只在Y轴添加，放在图层底部 zorder=0)
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 在柱子上方添加具体的数值标签 (Mean ± Std)
def autolabel(rects, stds):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect, std in zip(rects, stds):
        height = rect.get_height()
        # 标注文本，放在误差棒的上方
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height + std + 0.02),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1, transformer_stds)
autolabel(rects2, rf_stds)

# 隐藏上边框和右边框，让图表看起来更干净
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ==========================================
# 5. 保存图表
# ==========================================
plt.tight_layout()
plt.savefig('generalizability_comparison_kaggle.png', dpi=300, bbox_inches='tight')
plt.savefig('generalizability_comparison_kaggle.pdf', format='pdf', bbox_inches='tight')

print("✓ 成功生成高分辨率对比图！已保存为 PNG 和 PDF 格式。")
# plt.show() # 如果你在 Jupyter 或本地环境运行，可以取消注释预览
