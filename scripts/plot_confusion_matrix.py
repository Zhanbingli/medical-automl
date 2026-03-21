import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置 SCI 字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def load_json(filename):
    if not Path(filename).exists():
        print(f"Error: {filename} not found.")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_cm_heatmap():
    internal_data = load_json('results_kfold_5.json')
    external_data = load_json('external_validation_results.json')

    if not internal_data or not external_data:
        return

    # ---------------------------------------------------------
    # 1. 计算内部验证 (UCI) 的混淆矩阵
    # ---------------------------------------------------------
    all_y_true = []
    all_y_pred = []
    for fold in internal_data['individual_results']:
        y_true = fold['y_true']
        # 使用 0.5 作为默认阈值来获取预测类别
        y_pred = [1 if p >= 0.5 else 0 for p in fold['y_prob']]
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    tp_int = np.sum((all_y_true == 1) & (all_y_pred == 1))
    fn_int = np.sum((all_y_true == 1) & (all_y_pred == 0))
    fp_int = np.sum((all_y_true == 0) & (all_y_pred == 1))
    tn_int = np.sum((all_y_true == 0) & (all_y_pred == 0))

    cm_internal = np.array([[tn_int, fp_int],
                            [fn_int, tp_int]])
    cm_internal_pct = cm_internal / np.sum(cm_internal) * 100

    # ---------------------------------------------------------
    # 2. 重构外部验证 (Kaggle) 的平均混淆矩阵
    # ---------------------------------------------------------
    # Kaggle Heart Failure 数据集固定分布: Positive=508, Negative=410, Total=918
    P_ext = 508
    N_ext = 410

    mean_sens = external_data['summary']['sensitivity']['mean']
    mean_spec = external_data['summary']['specificity']['mean']

    tp_ext = int(round(mean_sens * P_ext))
    fn_ext = P_ext - tp_ext
    tn_ext = int(round(mean_spec * N_ext))
    fp_ext = N_ext - tn_ext

    cm_external = np.array([[tn_ext, fp_ext],
                            [fn_ext, tp_ext]])
    cm_external_pct = cm_external / np.sum(cm_external) * 100

    # ---------------------------------------------------------
    # 3. 绘制并排热图
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    labels_text = [['True Negative (TN)', 'False Positive (FP)\n(Type I Error)'],
                   ['False Negative (FN)\n(Type II Error)', 'True Positive (TP)']]

    def plot_heatmap(ax, cm, cm_pct, title, cmap):
        # 构造单元格内显示的文本 (数量 + 百分比)
        annot = np.empty_like(cm, dtype=object)
        for i in range(2):
            for j in range(2):
                annot[i, j] = f"{labels_text[i][j]}\n\n{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

        sns.heatmap(cm_pct, annot=annot, fmt='', cmap=cmap, cbar=False,
                    ax=ax, annot_kws={"size": 12, "weight": "bold"},
                    linewidths=2, linecolor='black')

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Predicted Diagnosis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual Diagnosis', fontsize=14, fontweight='bold')
        ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
        ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12, va='center')

    # 左图：内部验证 (蓝色系)
    plot_heatmap(axes[0], cm_internal, cm_internal_pct,
                 f'Internal Validation (UCI, n={np.sum(cm_internal)})', 'Blues')

    # 右图：外部验证 (橙红色系)
    plot_heatmap(axes[1], cm_external, cm_external_pct,
                 f'External Validation (Kaggle, n={np.sum(cm_external)})', 'OrRd')

    plt.suptitle('Prediction Error Analysis: Generalization Collapse Patterns',
                 fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()

    plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('confusion_matrix_comparison.pdf', format='pdf', bbox_inches='tight')
    print("✓ 成功生成混淆矩阵对比热图：confusion_matrix_comparison.png")

if __name__ == "__main__":
    generate_cm_heatmap()
