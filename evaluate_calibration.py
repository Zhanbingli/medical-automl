#!/usr/bin/env python3
"""
Medical AutoML - Calibration Analysis for Transformer (Direct JSON Loader)

Evaluates probability calibration of the Transformer model using 5-fold CV.
Reads true labels and predicted probabilities directly from the saved JSON
results to ensure 100% consistency with training logs.
Calculates Brier Score, Expected Calibration Error (ECE), and plots
calibration curves with confidence intervals.

Usage: python evaluate_calibration.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Configuration
K_FOLDS = 5
N_BINS = 10

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_prob[in_bin])
            ece += (bin_size / len(y_true)) * np.abs(bin_accuracy - bin_confidence)

    return ece

def plot_calibration_detailed(fold_y_true, fold_y_prob):
    """Plot detailed calibration analysis."""
    # Set style for SCI
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = (14, 12)
    plt.rcParams['font.size'] = 11

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

    # Calculate metrics
    all_true = np.concatenate(fold_y_true)
    all_prob = np.concatenate(fold_y_prob)

    brier_scores = [brier_score_loss(yt, yp) for yt, yp in zip(fold_y_true, fold_y_prob)]
    ece_scores = [calculate_ece(yt, yp, N_BINS) for yt, yp in zip(fold_y_true, fold_y_prob)]

    mean_brier = np.mean(brier_scores)
    std_brier = np.std(brier_scores, ddof=1)
    mean_ece = np.mean(ece_scores)
    std_ece = np.std(ece_scores, ddof=1)

    # ============== Plot 1: Main Calibration Curve ==============
    ax1 = axes[0, 0]

    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly calibrated', zorder=1)

    prob_true_list = []
    prob_pred_list = []

    for i, (y_true, y_prob) in enumerate(zip(fold_y_true, fold_y_prob)):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy='uniform')
        prob_true_list.append(prob_true)
        prob_pred_list.append(prob_pred)

        # Individual fold curves
        ax1.plot(prob_pred, prob_true, 'o-', alpha=0.3, linewidth=1,
                 color='gray', label=f'Fold {i+1}' if i == 0 else '')

    # Interpolate for mean curve calculation
    all_pairs = []
    for i in range(K_FOLDS):
        for j in range(len(prob_pred_list[i])):
            all_pairs.append((prob_true_list[i][j], prob_pred_list[i][j]))

    all_pairs.sort(key=lambda x: x[0])
    true_vals = [p[0] for p in all_pairs]
    pred_vals = [p[1] for p in all_pairs]

    bin_edges = np.linspace(0, 1, N_BINS + 1)
    mean_pred_binned = []
    mean_true_binned = []
    std_pred_binned = []

    for b in range(N_BINS):
        in_bin = (np.array(true_vals) >= bin_edges[b]) & (np.array(true_vals) < bin_edges[b+1])
        if np.sum(in_bin) > 0:
            mean_pred_binned.append(np.mean(np.array(pred_vals)[in_bin]))
            mean_true_binned.append((bin_edges[b] + bin_edges[b+1]) / 2)
            std_pred_binned.append(np.std(np.array(pred_vals)[in_bin]))

    ax1.plot(mean_pred_binned, mean_true_binned, 'o-', color='#2E86AB',
             linewidth=3, markersize=8, label=f'Mean (n={K_FOLDS})', zorder=2)

    ax1.fill_between(mean_true_binned,
                     [m - s for m, s in zip(mean_pred_binned, std_pred_binned)],
                     [m + s for m, s in zip(mean_pred_binned, std_pred_binned)],
                     alpha=0.2, color='#2E86AB', label='±1 std')

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax1.set_title('Calibration Curve (Reliability Diagram)\nTransformer Model',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # ============== Plot 2: Prediction Distribution ==============
    ax2 = axes[0, 1]

    pos_probs = all_prob[all_true == 1]
    neg_probs = all_prob[all_true == 0]

    ax2.hist(neg_probs, bins=20, alpha=0.6, label=f'Negative (n={len(neg_probs)})',
             color='#A23B72', edgecolor='black', linewidth=0.5)
    ax2.hist(pos_probs, bins=20, alpha=0.6, label=f'Positive (n={len(pos_probs)})',
             color='#2E86AB', edgecolor='black', linewidth=0.5)

    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    ax2.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Prediction Distribution\nby True Class', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper center', fontsize=10)

    # ============== Plot 3: Per-Fold Calibration ==============
    ax3 = axes[1, 0]
    fold_nums = list(range(1, K_FOLDS + 1))

    ax3.bar([f - 0.2 for f in fold_nums], brier_scores, 0.35,
             label='Brier Score', color='#2E86AB', alpha=0.8, edgecolor='black')
    ax3.axhline(y=mean_brier, color='#2E86AB', linestyle='--', linewidth=2,
                 label=f'Mean: {mean_brier:.4f}')

    ax3.bar([f + 0.2 for f in fold_nums], ece_scores, 0.35,
             label='ECE', color='#F18F01', alpha=0.8, edgecolor='black')
    ax3.axhline(y=mean_ece, color='#F18F01', linestyle='--', linewidth=2,
                 label=f'Mean: {mean_ece:.4f}')

    ax3.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Fold Calibration Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(fold_nums)
    ax3.legend(loc='upper right', fontsize=10)

    # ============== Plot 4: Summary Metrics ==============
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    ═══════════════════════════════════════════════════
                    CALIBRATION SUMMARY
    ═══════════════════════════════════════════════════

    Brier Score (lower is better):
        • Mean: {mean_brier:.4f} ± {std_brier:.4f}
        • Range: [{min(brier_scores):.4f}, {max(brier_scores):.4f}]

    Expected Calibration Error (ECE) (lower is better):
        • Mean: {mean_ece:.4f} ± {std_ece:.4f}
        • Range: [{min(ece_scores):.4f}, {max(ece_scores):.4f}]

    Total Samples: {len(all_true)}

    ═══════════════════════════════════════════════════
                         INTERPRETATION
    ═══════════════════════════════════════════════════

    {'✅ WELL-CALIBRATED' if mean_ece < 0.05 else '⚠️ NEEDS CALIBRATION'}
    ECE < 0.05 indicates good calibration

    Brier Score < 0.25 indicates useful model
    (theoretical minimum: 0.0 for perfect predictions)

    ═══════════════════════════════════════════════════
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Transformer Model Calibration Analysis (5-Fold CV)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("读取准确的 JSON 数据进行校准分析...")

    with open('results_kfold_5.json', 'r') as f:
        data = json.load(f)

    results = data['individual_results']

    fold_y_true = []
    fold_y_prob = []

    print("\n" + "=" * 50)
    print("每折校准指标 (直接从真实训练结果计算):")
    print("=" * 50)

    for fold_data in results:
        y_true = np.array(fold_data['y_true'])
        y_prob = np.array(fold_data['y_prob'])

        fold_y_true.append(y_true)
        fold_y_prob.append(y_prob)

        brier = brier_score_loss(y_true, y_prob)
        ece = calculate_ece(y_true, y_prob, N_BINS)
        print(f"Fold {fold_data['fold']}: Brier={brier:.4f}, ECE={ece:.4f}")

    print("\n生成并保存 SCI 级别校准图表...")
    fig = plot_calibration_detailed(fold_y_true, fold_y_prob)

    output_png = 'calibration_analysis_perfect_match.png'
    output_pdf = 'calibration_analysis_perfect_match.pdf'

    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')

    print(f"✅ 图表已保存为: {output_pdf}")

if __name__ == "__main__":
    main()
