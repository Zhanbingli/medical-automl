import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap

# 1. 准备数据
df = pd.read_csv('results_clinical.tsv', sep='\t')
df['version_idx'] = range(1, len(df) + 1) # 版本序号

# 2. 样式设置 (SCI 风格)
plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
plt.subplots_adjust(hspace=0.05) # 极窄的间距使上下图产生融合感

# --- 上半部分：性能指标 (Line + Scatter) ---
# 绘制趋势线
ax1.plot(df['version_idx'], df['val_auc'], color='#2E5A88', linewidth=2.5, label='Validation AUC', zorder=1)
ax1.plot(df['version_idx'], df['val_acc'], color='#D97B29', linewidth=2, linestyle='--', label='Validation Acc', zorder=1)

# 分类标记点
keep = df[df['status'] == 'keep']
discard = df[df['status'] == 'discard']

ax1.scatter(keep['version_idx'], keep['val_auc'], color='#2E5A88', s=100, marker='o', edgecolor='black', label='Selected Model', zorder=3)
ax1.scatter(discard['version_idx'], discard['val_auc'], color='white', s=100, marker='o', edgecolor='#2E5A88', label='Discarded Iteration', zorder=3)

# 标注序号
for idx, row in df.iterrows():
    ax1.annotate(str(idx+1), (row['version_idx'], row['val_auc']), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

ax1.set_ylabel('Performance Score', fontsize=12)
ax1.set_ylim(0.5, 1.05)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', frameon=True)
ax1.set_title('Model Evolution & Structural Optimization', loc='left', fontsize=14, fontweight='bold')

# --- 下半部分：架构修改描述 (Step-wise Bars) ---
# 使用不同颜色的色块代表 Keep/Discard 状态
colors = ['#C6E2FF' if s == 'keep' else '#FFC1C1' for s in df['status']]
ax2.bar(df['version_idx'], [1]*len(df), color=colors, edgecolor='black', alpha=0.7, width=0.8)

# 在色块中自动换行填写描述文字
for i, desc in enumerate(df['description']):
    short_desc = desc.split('-')[0].strip() # 提取主要描述
    wrapped = textwrap.fill(short_desc, width=18)
    ax2.text(i+1, 0.5, wrapped, ha='center', va='center', fontsize=8)

ax2.set_yticks([]) # 隐藏 Y 轴
ax2.set_xlabel('Iteration Step (Model Version)', fontsize=12)
ax2.set_xticks(df['version_idx'])
ax2.set_xticklabels([f"V{i}" for i in df['version_idx']])

# 添加下层图例
keep_patch = mpatches.Patch(color='#C6E2FF', label='Architecture Accepted')
discard_patch = mpatches.Patch(color='#FFC1C1', label='Architecture Rejected')
ax2.legend(handles=[keep_patch, discard_patch], loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('fusion_plot_sci.png', dpi=300, bbox_inches='tight')
print("SCI 融合画法图表已生成：fusion_plot_sci.png")
