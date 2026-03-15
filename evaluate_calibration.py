#!/usr/bin/env python3
"""
Medical AutoML - Calibration Analysis for Transformer

Evaluates probability calibration of the Transformer model using 5-fold CV.
Calculates Brier Score, Expected Calibration Error (ECE), and plots 
calibration curves with confidence intervals.

Usage: uv run python evaluate_calibration.py

Requires: saved_models/model_fold{0-4}.pt and data/val_fold{0-4}.bin
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve, brier_score_loss
from sklearn.metrics import roc_auc_score

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available.")
    print("Environment verified: macOS with Metal acceleration")
    print()

verify_macos_env()

# Import from project
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, DATA_DIR, bos_token_id
from train import GPTConfig, GPT, build_model_config

# Configuration
K_FOLDS = 5
N_BINS = 10
RANDOM_STATE = 42

# Hyperparameters (must match training)
from train import (
    ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN,
    TOTAL_BATCH_SIZE, EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR,
    WEIGHT_DECAY, ADAM_BETAS, WARMUP_RATIO, WARMDOWN_RATIO,
    FINAL_LR_FRAC, DROPOUT, DEPTH, DEVICE_BATCH_SIZE
)

# Device setup
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)
print(f"Using device: {device_type}")

if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
else:
    import contextlib
    autocast_ctx = contextlib.nullcontext()


def get_model_predictions(model, tokenizer, split_name):
    """Get predictions for all samples in validation set."""
    device = next(model.parameters()).device
    
    if "_fold" in split_name:
        val_path = os.path.join(DATA_DIR, f"{split_name}.bin")
    else:
        val_path = os.path.join(DATA_DIR, "val.bin")
    
    val_tokens = np.fromfile(val_path, dtype=np.int32).tolist()
    
    # Get token ID for '1' (disease presence)
    try:
        token_id_1 = tokenizer.enc.encode_single_token('1')
    except:
        token_id_1 = tokenizer.enc.encode_ordinary('1')[-1]
    
    y_true = []
    y_prob = []
    
    current_doc = []
    for token_id in val_tokens:
        if token_id == bos_token_id and current_doc:
            if len(current_doc) >= 2:
                last_token = current_doc[-1]
                context = current_doc[:-1]
                
                if len(context) > MAX_SEQ_LEN:
                    context = context[-MAX_SEQ_LEN:]
                
                input_tensor = torch.tensor([context], dtype=torch.long, device=device)
                
                with autocast_ctx:
                    logits = model(input_tensor)
                
                probs = F.softmax(logits[0, -1], dim=-1)
                prob_1 = probs[token_id_1].item()
                
                actual_label = 1 if last_token == token_id_1 else 0
                
                y_true.append(actual_label)
                y_prob.append(prob_1)
            
            current_doc = [token_id]
        else:
            current_doc.append(token_id)
    
    return np.array(y_true), np.array(y_prob)


def load_model_and_predict(fold_idx, tokenizer):
    """Load a trained model and get predictions on its validation set."""
    print(f"  Loading model_fold{fold_idx}.pt...")
    
    # Build model
    config = build_model_config(DEPTH)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    
    # Load weights
    model_path = f"saved_models/model_fold{fold_idx}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get predictions
    y_true, y_prob = get_model_predictions(model, tokenizer, f"val_fold{fold_idx}")
    
    # Cleanup
    del model
    gc.collect()
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()
    
    return y_true, y_prob


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


def plot_calibration_detailed(all_y_true, all_y_prob, fold_y_true, fold_y_prob):
    """Plot detailed calibration analysis."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 12)
    plt.rcParams['font.size'] = 11
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate overall metrics
    all_true = np.concatenate(all_y_true)
    all_prob = np.concatenate(all_y_prob)
    
    overall_brier = brier_score_loss(all_true, all_prob)
    
    # Calculate per-fold metrics
    brier_scores = [brier_score_loss(yt, yp) for yt, yp in zip(fold_y_true, fold_y_prob)]
    ece_scores = [calculate_ece(yt, yp, N_BINS) for yt, yp in zip(fold_y_true, fold_y_prob)]
    
    mean_brier = np.mean(brier_scores)
    std_brier = np.std(brier_scores)
    mean_ece = np.mean(ece_scores)
    std_ece = np.std(ece_scores)
    
    # ============== Plot 1: Main Calibration Curve ==============
    ax1 = axes[0, 0]
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly calibrated', zorder=1)
    
    # Calculate calibration curve for each fold
    prob_true_list = []
    prob_pred_list = []
    
    for i, (y_true, y_prob) in enumerate(zip(fold_y_true, fold_y_prob)):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=N_BINS, strategy='uniform')
        prob_true_list.append(prob_true)
        prob_pred_list.append(prob_pred)
        
        # Individual fold curves (dashed, semi-transparent)
        ax1.plot(prob_pred, prob_true, 'o-', alpha=0.3, linewidth=1, 
                 color='gray', label=f'Fold {i+1}' if i == 0 else '')
    
    # Calculate mean curve
    max_bins = max(len(pt) for pt in prob_true_list)
    mean_true = []
    mean_pred = []
    std_true = []
    
    for b in range(max_bins):
        vals = [prob_true_list[i][b] for i in range(K_FOLDS) if b < len(prob_true_list[i])]
        if vals:
            mean_true.append(np.mean(vals))
            std_true.append(np.std(vals))
    
    # Interpolate to same points for averaging
    from scipy.interpolate import interp1d
    
    # Mean calibration curve (average of predictions at same true fractions)
    all_pairs = []
    for i in range(K_FOLDS):
        for j in range(len(prob_pred_list[i])):
            all_pairs.append((prob_true_list[i][j], prob_pred_list[i][j]))
    
    all_pairs.sort(key=lambda x: x[0])
    true_vals = [p[0] for p in all_pairs]
    pred_vals = [p[1] for p in all_pairs]
    
    # Bin and average
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
    
    # Add confidence band (shaded area)
    # Use prediction fraction as uncertainty measure
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
    ax1.grid(True, alpha=0.3)
    
    # ============== Plot 2: Prediction Distribution ==============
    ax2 = axes[0, 1]
    
    # Separate positive and negative samples
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
    ax2.grid(True, alpha=0.3)
    
    # ============== Plot 3: Per-Fold Calibration ==============
    ax3 = axes[1, 0]
    
    fold_nums = list(range(1, K_FOLDS + 1))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, K_FOLDS))
    
    # Brier score (lower is better)
    bars1 = ax3.bar([f - 0.2 for f in fold_nums], brier_scores, 0.35, 
                     label='Brier Score', color='#2E86AB', alpha=0.8, edgecolor='black')
    ax3.axhline(y=mean_brier, color='#2E86AB', linestyle='--', linewidth=2, 
                 label=f'Mean: {mean_brier:.4f}')
    
    # ECE (lower is better)
    bars2 = ax3.bar([f + 0.2 for f in fold_nums], ece_scores, 0.35,
                     label='ECE', color='#F18F01', alpha=0.8, edgecolor='black')
    ax3.axhline(y=mean_ece, color='#F18F01', linestyle='--', linewidth=2,
                 label=f'Mean: {mean_ece:.4f}')
    
    ax3.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Fold Calibration Metrics', fontsize=14, fontweight='bold')
    ax3.set_xticks(fold_nums)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============== Plot 4: Summary Metrics ==============
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
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
    
    return fig, {
        'brier_score': {'mean': mean_brier, 'std': std_brier, 'per_fold': brier_scores},
        'ece': {'mean': mean_ece, 'std': std_ece, 'per_fold': ece_scores},
        'n_samples': len(all_true)
    }


def main():
    print("=" * 70)
    print("Medical AutoML - Transformer Calibration Analysis")
    print("=" * 70)
    
    # Check model files exist
    for i in range(K_FOLDS):
        model_path = f"saved_models/model_fold{i}.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            print("Please run train_kfold.py first to generate models.")
            return
    
    print(f"\nFound {K_FOLDS} model files")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")
    
    # Collect predictions from all folds
    print("\nLoading models and collecting predictions...")
    fold_y_true = []
    fold_y_prob = []
    fold_metrics = []
    
    for fold_idx in range(K_FOLDS):
        y_true, y_prob = load_model_and_predict(fold_idx, tokenizer)
        
        brier = brier_score_loss(y_true, y_prob)
        ece = calculate_ece(y_true, y_prob, N_BINS)
        
        fold_y_true.append(y_true)
        fold_y_prob.append(y_prob)
        fold_metrics.append({
            'fold': fold_idx + 1,
            'n_samples': len(y_true),
            'brier_score': brier,
            'ece': ece
        })
        
        print(f"  Fold {fold_idx + 1}: {len(y_true)} samples, "
              f"Brier={brier:.4f}, ECE={ece:.4f}")
    
    # Calculate overall metrics
    all_y_true = np.concatenate(fold_y_true)
    all_y_prob = np.concatenate(fold_y_prob)
    
    overall_brier = brier_score_loss(all_y_true, all_y_prob)
    overall_ece = calculate_ece(all_y_true, all_y_prob, N_BINS)
    
    mean_brier = np.mean([m['brier_score'] for m in fold_metrics])
    std_brier = np.std([m['brier_score'] for m in fold_metrics])
    mean_ece = np.mean([m['ece'] for m in fold_metrics])
    std_ece = np.std([m['ece'] for m in fold_metrics])
    
    # Print summary
    print("\n" + "=" * 70)
    print("CALIBRATION METRICS SUMMARY")
    print("=" * 70)
    
    print(f"\nBrier Score (lower is better):")
    print(f"  Overall: {overall_brier:.4f}")
    print(f"  Mean ± Std: {mean_brier:.4f} ± {std_brier:.4f}")
    print(f"  Per-fold: {[f'{m['brier_score']:.4f}' for m in fold_metrics]}")
    
    print(f"\nExpected Calibration Error (ECE) (lower is better):")
    print(f"  Overall: {overall_ece:.4f}")
    print(f"  Mean ± Std: {mean_ece:.4f} ± {std_ece:.4f}")
    print(f"  Per-fold: {[f'{m['ece']:.4f}' for m in fold_metrics]}")
    
    print(f"\nTotal validation samples: {len(all_y_true)}")
    print(f"Positive: {np.sum(all_y_true)}, Negative: {len(all_y_true) - np.sum(all_y_true)}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if mean_ece < 0.05:
        print("✅ WELL-CALIBRATED: ECE < 0.05")
    elif mean_ece < 0.10:
        print("⚠️ MODERATELY CALIBRATED: 0.05 ≤ ECE < 0.10")
    else:
        print("❌ POORLY CALIBRATED: ECE ≥ 0.10")
    
    if overall_brier < 0.25:
        print("✅ USEFUL MODEL: Brier Score < 0.25")
    else:
        print("⚠️ HIGH Brier Score: Model may have limited utility")
    
    # Save metrics to JSON
    results = {
        'model': 'Transformer',
        'n_folds': K_FOLDS,
        'brier_score': {
            'overall': overall_brier,
            'mean': mean_brier,
            'std': std_brier,
            'per_fold': [m['brier_score'] for m in fold_metrics]
        },
        'ece': {
            'overall': overall_ece,
            'mean': mean_ece,
            'std': std_ece,
            'per_fold': [m['ece'] for m in fold_metrics]
        },
        'n_samples': len(all_y_true),
        'n_bins': N_BINS
    }
    
    with open('calibration_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: calibration_metrics.json")
    
    # Generate plots
    print("\nGenerating calibration plots...")
    fig, metrics = plot_calibration_detailed(fold_y_true, fold_y_prob, 
                                              fold_y_true, fold_y_prob)
    
    output_path = 'calibration_analysis.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    plt.close()
    
    print("\n" + "=" * 70)
    print("✅ Calibration analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
