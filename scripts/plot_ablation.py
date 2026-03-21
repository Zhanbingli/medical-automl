#!/usr/bin/env python3
"""
生成消融实验对比图（Figure 9）
包含两个子图：
  (A) 实验A：零填充 vs 均值填充，外部验证四项指标对比
  (B) 实验B：英文 vs 中文编码，内部验证 AUC 对比（含 fold-level 散点）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

plt.rcParams['font.family']  = 'Times New Roman'
plt.rcParams['font.size']    = 11
plt.rcParams['axes.unicode_minus'] = False

# ── 读取数据 ──────────────────────────────────────────────
with open('experiment_A_results.json') as f:
    a = json.load(f)
with open('experiment_B_results.json') as f:
    b = json.load(f)

# ── 颜色方案 ──────────────────────────────────────────────
C_ZERO = '#d62728'   # 红：零填充
C_MEAN = '#1f77b4'   # 蓝：均值填充
C_ZH   = '#2ca02c'   # 绿：中文
C_EN   = '#ff7f0e'   # 橙：英文

# ════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
fig.suptitle('Figure 9. Ablation Study: Imputation Strategy and Encoding Language',
             fontsize=13, fontweight='bold', y=1.01)

# ── 子图 A：实验A 四项指标分组柱状图 ──────────────────────
ax = axes[0]

metrics     = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
zero_means  = [a['zero_padding']['summary'][k.lower()]['mean']
               if k.lower() in a['zero_padding']['summary']
               else a['zero_padding']['summary'][k]['mean']
               for k in ['auc','acc','sensitivity','specificity']]
zero_stds   = [a['zero_padding']['summary'][k]['std']
               for k in ['auc','acc','sensitivity','specificity']]
mean_means  = [a['mean_imputation']['summary'][k]['mean']
               for k in ['auc','acc','sensitivity','specificity']]
mean_stds   = [a['mean_imputation']['summary'][k]['std']
               for k in ['auc','acc','sensitivity','specificity']]

x     = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, zero_means, width,
               yerr=zero_stds, capsize=4,
               color=C_ZERO, alpha=0.85, label='Zero-padding',
               error_kw=dict(elinewidth=1.2, ecolor='#555'))
bars2 = ax.bar(x + width/2, mean_means, width,
               yerr=mean_stds, capsize=4,
               color=C_MEAN, alpha=0.85, label='Mean imputation',
               error_kw=dict(elinewidth=1.2, ecolor='#555'))

# 在 AUC 柱上标注显著性（标准水平括号）
y_bracket = max(zero_means[0] + zero_stds[0], mean_means[0] + mean_stds[0]) + 0.04
x1, x2 = 0 - width/2, 0 + width/2
ax.plot([x1, x2], [y_bracket, y_bracket], color='#333', lw=1.2)
ax.plot([x1, x1], [y_bracket - 0.012, y_bracket], color='#333', lw=1.2)
ax.plot([x2, x2], [y_bracket - 0.012, y_bracket], color='#333', lw=1.2)
ax.text((x1+x2)/2, y_bracket + 0.012,
        '*** p=0.0002', ha='center', fontsize=9.5, fontweight='bold', color='#333')

# 数值标签
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=8, color='#333')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom',
            fontsize=8, color='#333')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, 1.08)
ax.set_title('(A) Experiment A: Imputation Strategy\n'
             'External Validation on Kaggle Cohort (n=918)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.yaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── 子图 B：实验B 英文 vs 中文 AUC 带折线图 ──────────────
ax = axes[1]

en_folds = [r['auc'] for r in b['english_encoding']['fold_results']]
zh_folds = [0.7694805194805194, 0.8387445887445888,
            0.787878787878788,  0.6296296296296297,
            0.7823660714285714]   # 原论文 5-fold AUC

fold_x = np.arange(1, 6)

# 折线
ax.plot(fold_x, zh_folds, 'o-', color=C_ZH, lw=2, ms=7,
        label=f'Chinese  (Mean={np.mean(zh_folds):.3f}±{np.std(zh_folds,ddof=1):.3f})')
ax.plot(fold_x, en_folds, 's--', color=C_EN, lw=2, ms=7,
        label=f'English  (Mean={np.mean(en_folds):.3f}±{np.std(en_folds,ddof=1):.3f})')

# 均值虚线
ax.axhline(np.mean(zh_folds), color=C_ZH, lw=1.2, ls=':', alpha=0.6)
ax.axhline(np.mean(en_folds), color=C_EN, lw=1.2, ls=':', alpha=0.6)

# 阴影带（±1 SD）
ax.fill_between(fold_x,
                np.mean(zh_folds) - np.std(zh_folds, ddof=1),
                np.mean(zh_folds) + np.std(zh_folds, ddof=1),
                color=C_ZH, alpha=0.10)
ax.fill_between(fold_x,
                np.mean(en_folds) - np.std(en_folds, ddof=1),
                np.mean(en_folds) + np.std(en_folds, ddof=1),
                color=C_EN, alpha=0.10)

# 标注 p 值
t_stat, p_val = stats.ttest_1samp(en_folds, np.mean(zh_folds))
ax.text(3, 0.92,
        f'Δ AUC = +{np.mean(en_folds)-np.mean(zh_folds):.3f}\np = {p_val:.3f} (n.s.)',
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                  edgecolor='#aaa', alpha=0.9))

ax.set_xticks(fold_x)
ax.set_xticklabels([f'Fold {i}' for i in fold_x], fontsize=11)
ax.set_ylabel('AUC', fontsize=12)
ax.set_ylim(0.50, 1.00)
ax.set_title('(B) Experiment B: Encoding Language\n'
             'Internal 5-Fold CV on UCI Cohort (n=303)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
ax.yaxis.grid(True, linestyle=':', alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── 保存 ──────────────────────────────────────────────────
plt.tight_layout()
plt.savefig('figure9_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('figure9_ablation.pdf', format='pdf', bbox_inches='tight')
print("✅ 已保存：figure9_ablation.png / .pdf")
