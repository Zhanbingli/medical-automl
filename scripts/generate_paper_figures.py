import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置 SCI 标准字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def load_json(filename):
    if not Path(filename).exists():
        print(f"Error: {filename} not found.")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_combined_figure():
    # 1. 加载数据
    internal_data = load_json('results_kfold_5.json')
    external_data = load_json('external_validation_results.json')
    stats_data = load_json('statistical_tests_results.json')

    if not (internal_data and external_data and stats_data):
        return

    # 创建大图，包含左右两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # ---------------------------------------------------------
    # Panel A: Generalization Gap (Internal vs. External)
    # ---------------------------------------------------------
    # 定义两个文件之间不同的键名映射
    # 内部数据 (results_kfold_5.json) 使用缩写
    internal_keys = ['val_auc', 'val_acc', 'val_sens', 'val_spec']
    # 外部数据 (external_validation_results.json) 使用全称
    external_keys = ['auc', 'acc', 'sensitivity', 'specificity']

    metric_labels = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']

    # 提取内部数据均值和标准差
    internal_means = [internal_data['summary'][k]['mean'] for k in internal_keys]
    internal_stds = [internal_data['summary'][k]['std'] for k in internal_keys]

    # 提取外部数据均值和标准差
    external_means = [external_data['summary'][k]['mean'] for k in external_keys]
    external_stds = [external_data['summary'][k]['std'] for k in external_keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    ax1.bar(x - width/2, internal_means, width, yerr=internal_stds,
            label='Internal Validation (UCI)', color='#d62728', alpha=0.8, capsize=5, edgecolor='black')
    ax1.bar(x + width/2, external_means, width, yerr=external_stds,
            label='External Validation (Kaggle)', color='#ff7f0e', alpha=0.6, capsize=5, edgecolor='black')

    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('A. Generalization Performance Gap', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # ---------------------------------------------------------
    # Panel B: Statistical Significance Forest Plot
    # ---------------------------------------------------------
    comparisons = stats_data['comparisons']
    models = list(comparisons.keys())
    mean_diffs = [comparisons[m]['mean_difference'] for m in models]
    ci_low = [comparisons[m]['confidence_interval_95']['lower'] for m in models]
    ci_high = [comparisons[m]['confidence_interval_95']['upper'] for m in models]

    errors = [np.array(mean_diffs) - np.array(ci_low), np.array(ci_high) - np.array(mean_diffs)]
    y_pos = np.arange(len(models))

    ax2.errorbar(mean_diffs, y_pos, xerr=errors, fmt='o', color='#1f77b4',
                 markersize=8, capsize=6, elinewidth=2, markeredgecolor='black')

    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=12)
    ax2.invert_yaxis()

    ax2.set_xlabel('Mean Difference in AUC (vs. Transformer)', fontsize=14, fontweight='bold')
    ax2.set_title('B. Paired t-test Results (95% CI)', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(axis='x', linestyle=':', alpha=0.6)

    # 调整布局并保存
    plt.tight_layout(pad=4.0)
    plt.savefig('figure3_combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_combined_analysis.pdf', format='pdf', bbox_inches='tight')
    print("✓ 已生成组合分析图：figure3_combined_analysis.png 和 .pdf")
    plt.show()

if __name__ == "__main__":
    generate_combined_figure()
