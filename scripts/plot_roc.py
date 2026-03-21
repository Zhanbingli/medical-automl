#!/usr/bin/env python3
"""
Medical AutoML - 5-Fold CV ROC Curve Plotter (Direct JSON Loader)
彻底放弃重新推理，直接读取训练时保存的真实概率进行完美绘图。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def main():
    print("读取完美的 JSON 数据...")
    # 读取你刚刚生成的包含 y_true 和 y_prob 的 JSON 文件
    with open('results_kfold_5.json', 'r') as f:
        data = json.load(f)

    results = data['individual_results']

    tprs = []
    aucs_list = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    for fold_data in results:
        fold_idx = fold_data['fold']
        # 直接拿底层传出来的真实标签和概率
        y_true = np.array(fold_data['y_true'])
        y_prob = np.array(fold_data['y_prob'])
        expected_auc = fold_data['val_auc']

        # 计算 ROC 和 AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        aucs_list.append(roc_auc_val)

        # 插值以便后续求平均
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # 绘制单折的灰色细线
        ax.plot(fpr, tpr, color='gray', alpha=0.3, lw=1)

        # 验证算出来的 AUC 和训练日志里的是否 100% 一致
        match = "✓" if abs(roc_auc_val - expected_auc) < 0.0001 else "⚠ mismatch"
        print(f"Fold {fold_idx}: computed={roc_auc_val:.4f}  expected={expected_auc:.4f}  {match}")

    # 聚合求平均
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs_list))       # 5个fold AUC直接平均，与正文一致
    std_auc = float(np.std(aucs_list, ddof=1)) # 与正文一致

    print(f"\nMean AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # 绘制平均 ROC 红色粗线
    ax.plot(mean_fpr, mean_tpr, color='#b30000', lw=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.3f} $\\pm$ {std_auc:.3f})')

    # 绘制标准差阴影带
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#ff9999', alpha=0.3, label=r'$\pm$ 1 std. dev.')

    # 绘制基线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Chance (AUC = 0.500)')

    # 设置坐标轴与图例
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title('Transformer Internal 5-Fold CV ROC Curve', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    # 输出 SCI 标准的矢量图和位图
    plt.savefig('roc_curve_paper_perfect_match.png', dpi=300, bbox_inches='tight')
    plt.savefig('roc_curve_paper_perfect_match.pdf', format='pdf', bbox_inches='tight')

    print("\n✅ ROC 曲线已保存为 roc_curve_paper_perfect_match.pdf (推荐用于 SCI 论文)")

if __name__ == "__main__":
    main()
