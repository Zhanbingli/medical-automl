import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 准备你的实验数据
data = {
    'iteration': [1, 2, 3, 4, 5, 6],
    'commit': ['8505f61', 'e9d02b3', '35b4abc', '0abd527', 'd6e0a14', '131632c'],
    'val_auc': [0.818627, 0.813725, 0.926471, 0.901961, 0.941176, 0.759804],
    'val_acc': [0.689655, 0.655172, 0.793103, 0.758621, 0.827586, 0.620690],
    'status': ['keep', 'discard', 'keep', 'discard', 'keep', 'discard'],
    'description': [
        'Baseline (depth=3, ratio=32)',
        'Deeper (depth=6)',
        'Wider (ratio=48)',
        'Too Wide (ratio=64)',
        'Best (dropout=0.2)',
        'Deeper (depth=4)'
    ]
}
df = pd.DataFrame(data)

# 2. 设置学术绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

# 3. 绘制 AUC 曲线 (主干线)
color_auc = '#d62728' # 红色
ax1.set_xlabel('Agent Search Iteration', fontweight='bold', fontsize=12)
ax1.set_ylabel('Validation AUC', color=color_auc, fontweight='bold', fontsize=12)

# 区分 keep 和 discard 的点
keep_mask = df['status'] == 'keep'
discard_mask = df['status'] == 'discard'

ax1.plot(df['iteration'], df['val_auc'], color=color_auc, linestyle='-', linewidth=2, zorder=1)
ax1.scatter(df['iteration'][keep_mask], df['val_auc'][keep_mask], color=color_auc, s=120, label='Accepted (Keep)', zorder=2)
ax1.scatter(df['iteration'][discard_mask], df['val_auc'][discard_mask], color='white', edgecolor=color_auc, s=120, marker='o', label='Rejected (Discard)', zorder=2)
ax1.tick_params(axis='y', labelcolor=color_auc)

# 4. 绘制 ACC 曲线 (辅助线，双 Y 轴)
ax2 = ax1.twinx()
color_acc = '#1f77b4' # 蓝色
ax2.set_ylabel('Validation Accuracy', color=color_acc, fontweight='bold', fontsize=12)
ax2.plot(df['iteration'], df['val_acc'], color=color_acc, linestyle='--', linewidth=1.5, alpha=0.7)
ax2.scatter(df['iteration'][keep_mask], df['val_acc'][keep_mask], color=color_acc, marker='s', s=60, alpha=0.8)
ax2.scatter(df['iteration'][discard_mask], df['val_acc'][discard_mask], color='white', edgecolor=color_acc, marker='s', s=60, alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color_acc)

# 5. 添加学术批注 (Annotations)
for i, txt in enumerate(df['description']):
    # 只标注 keep 的节点，避免图面太杂乱
    if df['status'][i] == 'keep':
        ax1.annotate(txt,
                     (df['iteration'][i], df['val_auc'][i]),
                     xytext=(0, 15), textcoords='offset points',
                     ha='center', va='bottom', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# 6. 图例与美化
ax1.set_title('Agent-Driven Neural Architecture Search Trajectory', fontweight='bold', fontsize=15, pad=20)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_xticks(df['iteration'])
ax1.legend(loc='lower right', frameon=True)

fig.tight_layout()
plt.savefig('agent_search_trajectory.png')
print("✅ SCI 级别寻优轨迹图已生成：agent_search_trajectory.png")
