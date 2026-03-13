"""
Medical AutoML - Baseline Comparison Visualization

Creates publication-ready comparison plots for baseline models.

Usage: uv run python visualize_baselines.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def load_results():
    """Load baseline comparison results."""
    if Path('baseline_comparison_results.json').exists():
        with open('baseline_comparison_results.json', 'r') as f:
            return json.load(f)
    else:
        print("Error: baseline_comparison_results.json not found!")
        print("Please run: uv run python run_baseline_sota.py")
        return None

def plot_auc_comparison(results):
    """Plot AUC comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(results.keys())
    aucs = [results[m]['auc'] for m in models]
    
    # Color by model type
    colors = []
    for model in models:
        if 'Transformer' in model or 'ResNet' in model or 'MLP' in model or 'TabNet' in model:
            colors.append('#2E86AB')  # Deep learning - blue
        else:
            colors.append('#A23B72')  # Traditional ML - purple
    
    bars = ax.barh(models, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{auc:.4f}', ha='left', va='center', fontweight='bold')
    
    # Add transformer reference line
    ax.axvline(x=0.910, color='#F18F01', linestyle='--', linewidth=2, label='Your Transformer (AUC=0.910)')
    
    ax.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison - AUC Score\n(Higher is Better)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 1.0)
    ax.legend(loc='lower right')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.8, label='Deep Learning Models'),
        Patch(facecolor='#A23B72', alpha=0.8, label='Traditional ML Models')
    ]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='#F18F01', linestyle='--', 
                                                     linewidth=2, label='Your Transformer (AUC=0.910)')], 
              loc='lower right')
    
    plt.tight_layout()
    plt.savefig('baseline_comparison_auc.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_comparison_auc.png")
    plt.close()

def plot_clinical_metrics_radar(results):
    """Create radar chart for clinical metrics."""
    from math import pi
    
    # Select top 5 models by AUC
    top_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:5]
    
    categories = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_models)))
    
    for i, (model_name, metrics) in enumerate(top_models):
        values = [
            metrics['accuracy'],
            metrics['auc'],
            metrics['sensitivity'],
            metrics['specificity']
        ]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.title('Clinical Metrics Comparison - Top 5 Models\n(Balanced performance across all metrics)', 
              fontsize=14, fontweight='bold', pad=30)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison_radar.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_comparison_radar.png")
    plt.close()

def plot_metrics_heatmap(results):
    """Create heatmap of all metrics."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data
    models = list(results.keys())
    metrics_names = ['accuracy', 'auc', 'sensitivity', 'specificity']
    
    data = np.array([[results[m][metric] for metric in metrics_names] for m in models])
    
    # Create heatmap
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn', 
                xticklabels=[m.capitalize() for m in metrics_names],
                yticklabels=models, ax=ax, cbar_kws={'label': 'Score'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Clinical Metrics Heatmap - All Models\n(Green = Better, Red = Worse)', 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('baseline_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_comparison_heatmap.png")
    plt.close()

def create_summary_table(results):
    """Create a formatted summary table."""
    # Sort by AUC
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BASELINE COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<12} {'AUC':<12} {'Sensitivity':<14} {'Specificity':<14}")
    print("-" * 100)
    
    for i, (model, metrics) in enumerate(sorted_results, 1):
        print(f"{i:<6} {model:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['auc']:<12.4f} {metrics['sensitivity']:<14.4f} "
              f"{metrics['specificity']:<14.4f}")
    
    print("=" * 100)
    
    # Statistical summary
    print("\nSTATISTICAL SUMMARY")
    print("-" * 100)
    aucs = [r['auc'] for r in results.values()]
    print(f"Best AUC:  {max(aucs):.4f}")
    print(f"Worst AUC: {min(aucs):.4f}")
    print(f"Mean AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Your Transformer: 0.910 ± 0.021 (from K-Fold CV)")
    print("=" * 100)

def main():
    print("=" * 70)
    print("Medical AutoML - Baseline Visualization")
    print("=" * 70)
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    print(f"\nLoaded results for {len(results)} models")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_auc_comparison(results)
    plot_clinical_metrics_radar(results)
    plot_metrics_heatmap(results)
    
    # Print summary table
    create_summary_table(results)
    
    print("\n✅ All visualizations saved successfully!")
    print("\nGenerated files:")
    print("  1. baseline_comparison_auc.png - Bar chart comparison")
    print("  2. baseline_comparison_radar.png - Multi-metric radar chart")
    print("  3. baseline_comparison_heatmap.png - Metrics heatmap")
    print("  4. baseline_comparison_results.json - Raw data")

if __name__ == "__main__":
    main()
