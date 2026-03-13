"""
Medical AutoML - 5-Fold CV Baseline Visualization

Creates publication-ready comparison plots for 5-fold CV baseline results.
Automatically loads both baseline and Transformer JSON files.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_all_results():
    """Load both baseline and transformer results."""
    baseline_file = Path('baseline_comparison_5fold.json')
    transformer_file = Path('results_kfold_5.json')

    if not baseline_file.exists():
        print("Error: baseline_comparison_5fold.json not found!")
        return None, None

    if not transformer_file.exists():
        print("Error: results_kfold_5.json not found!")
        return None, None

    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)

    with open(transformer_file, 'r') as f:
        transformer_data = json.load(f)

    return baseline_data, transformer_data

def plot_auc_with_errorbars(summary, trans_summary):
    """Plot AUC comparison with error bars (mean ± std)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(summary.keys())
    auc_means = [summary[m]['auc']['mean'] for m in models]
    auc_stds = [summary[m]['auc']['std'] for m in models]

    # Color by model type
    colors = []
    for model in models:
        if model in ['MLP (Deep)', 'ResNet (Tabular)', 'TabNet']:
            colors.append('#2E86AB')  # Deep learning - blue
        else:
            colors.append('#A23B72')  # Traditional ML - purple

    # Sort by AUC
    sorted_indices = np.argsort(auc_means)[::-1]
    models_sorted = [models[i] for i in sorted_indices]
    auc_means_sorted = [auc_means[i] for i in sorted_indices]
    auc_stds_sorted = [auc_stds[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    # Create bar plot with error bars
    x_pos = np.arange(len(models_sorted))
    bars = ax.bar(x_pos, auc_means_sorted, yerr=auc_stds_sorted,
                  color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=1.5,
                  capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})

    # Fetch actual Transformer values
    transformer_auc = trans_summary['val_auc']['mean']
    transformer_std = trans_summary['val_auc']['std']

    ax.axhline(y=transformer_auc, color='#F18F01', linestyle='--', linewidth=2.5,
               label=f'Your Transformer: {transformer_auc:.3f} ± {transformer_std:.3f}')
    ax.fill_between([-0.5, len(models_sorted)-0.5],
                    transformer_auc - transformer_std,
                    transformer_auc + transformer_std,
                    color='#F18F01', alpha=0.15)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('5-Fold Cross Validation: AUC Comparison (Mean ± Std)\n' +
                 'Your Transformer vs. SOTA Baselines',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.7, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.8, label='Deep Learning Models'),
        Patch(facecolor='#A23B72', alpha=0.8, label='Traditional ML Models'),
        plt.Line2D([0], [0], color='#F18F01', linestyle='--', linewidth=2.5,
                   label=f'Transformer (AUC={transformer_auc:.3f}±{transformer_std:.3f})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig('baseline_5fold_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_metrics_comparison(summary, trans_summary):
    """Plot comparison of all 4 metrics with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    metrics = ['auc', 'accuracy', 'sensitivity', 'specificity']
    metric_names = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']

    # Map from baseline names to transformer names
    trans_keys = {'auc': 'val_auc', 'accuracy': 'val_acc', 'sensitivity': 'val_sens', 'specificity': 'val_spec'}

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        models = list(summary.keys())
        means = [summary[m][metric]['mean'] for m in models]
        stds = [summary[m][metric]['std'] for m in models]

        colors = ['#2E86AB' if m in ['MLP (Deep)', 'ResNet (Tabular)', 'TabNet']
                  else '#A23B72' for m in models]

        sorted_indices = np.argsort(means)[::-1]
        models_sorted = [models[i] for i in sorted_indices]
        means_sorted = [means[i] for i in sorted_indices]
        stds_sorted = [stds[i] for i in sorted_indices]
        colors_sorted = [colors[i] for i in sorted_indices]

        x_pos = np.arange(len(models_sorted))
        ax.bar(x_pos, means_sorted, yerr=stds_sorted, color=colors_sorted,
               alpha=0.8, edgecolor='black', linewidth=1, capsize=3)

        # Actual Transformer values
        t_key = trans_keys[metric]
        trans_mean = trans_summary[t_key]['mean']
        trans_std = trans_summary[t_key]['std']

        ax.axhline(y=trans_mean, color='#F18F01', linestyle='--', linewidth=2, alpha=0.8)
        ax.fill_between([-0.5, len(models_sorted)-0.5],
                        trans_mean - trans_std, trans_mean + trans_std,
                        color='#F18F01', alpha=0.1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models_sorted, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('5-Fold Cross Validation: All Clinical Metrics (Mean ± Std)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('baseline_5fold_all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_consistency(base_results, trans_results):
    """Plot fold-by-fold consistency for top models."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Get top 5 baseline models by mean AUC
    model_aucs = {m: np.mean([r['auc'] for r in base_results[m]]) for m in base_results.keys()}
    top_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)[:5]
    top_model_names = [m[0] for m in top_models]

    x = np.arange(1, 6)  # 5 folds
    width = 0.15

    colors = plt.cm.Set3(np.linspace(0, 1, len(top_model_names)))

    for i, model_name in enumerate(top_model_names):
        aucs = [r['auc'] for r in base_results[model_name]]
        offset = (i - len(top_model_names)/2) * width
        ax.bar(x + offset, aucs, width, label=model_name, color=colors[i],
               edgecolor='black', linewidth=0.5)

    # Actual Transformer values per fold
    transformer_aucs = [fold['val_auc'] for fold in trans_results['individual_results']]

    ax.plot(x, transformer_aucs, 'o-', color='#F18F01', linewidth=2.5,
            markersize=8, label='Your Transformer', zorder=10)

    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Fold-by-Fold Consistency: Top 5 Baselines vs Transformer',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.75, 1.05)

    plt.tight_layout()
    plt.savefig('baseline_5fold_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(summary, trans_summary):
    """Create formatted summary table."""
    print("\n" + "=" * 110)
    print("5-FOLD CROSS VALIDATION RESULTS TABLE")
    print("=" * 110)
    print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<20} {'AUC':<20} {'Sensitivity':<20} {'Specificity':<20}")
    print("-" * 110)

    # Sort by AUC
    ranked = sorted(summary.items(), key=lambda x: x[1]['auc']['mean'], reverse=True)

    # Put Transformer at the top as a benchmark
    t_acc = f"{trans_summary['val_acc']['mean']:.4f}±{trans_summary['val_acc']['std']:.4f}"
    t_auc = f"{trans_summary['val_auc']['mean']:.4f}±{trans_summary['val_auc']['std']:.4f}"
    t_sens = f"{trans_summary['val_sens']['mean']:.4f}±{trans_summary['val_sens']['std']:.4f}"
    t_spec = f"{trans_summary['val_spec']['mean']:.4f}±{trans_summary['val_spec']['std']:.4f}"

    print(f"{'★':<6} {'Your Transformer':<25} {t_acc:<20} {t_auc:<20} {t_sens:<20} {t_spec:<20}")
    print("-" * 110)

    for i, (model, metrics) in enumerate(ranked, 1):
        acc_str = f"{metrics['accuracy']['mean']:.4f}±{metrics['accuracy']['std']:.4f}"
        auc_str = f"{metrics['auc']['mean']:.4f}±{metrics['auc']['std']:.4f}"
        sens_str = f"{metrics['sensitivity']['mean']:.4f}±{metrics['sensitivity']['std']:.4f}"
        spec_str = f"{metrics['specificity']['mean']:.4f}±{metrics['specificity']['std']:.4f}"
        print(f"{i:<6} {model:<25} {acc_str:<20} {auc_str:<20} {sens_str:<20} {spec_str:<20}")

    print("=" * 110)

def main():
    print("=" * 70)
    print("5-Fold CV Baseline Visualization")
    print("=" * 70)

    baseline_data, transformer_data = load_all_results()
    if baseline_data is None or transformer_data is None:
        return

    summary = baseline_data['summary']
    results = baseline_data['individual_results']

    trans_summary = transformer_data['summary']

    print(f"\nLoaded results for {len(summary)} baseline models and 1 Transformer model")

    print("\nGenerating visualizations...")
    plot_auc_with_errorbars(summary, trans_summary)
    plot_all_metrics_comparison(summary, trans_summary)
    plot_fold_consistency(results, transformer_data)

    create_summary_table(summary, trans_summary)

    print("\n" + "=" * 70)
    print("✅ All visualizations saved successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
