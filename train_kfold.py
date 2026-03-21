"""
Medical AutoML - K-Fold Cross Validation Training

Performs K-fold cross validation for stable model evaluation.
Trains K models on different data splits and reports mean ± std metrics.

Part of the medical-automl project: https://github.com/Zhanbingli/medical-automl

Usage: uv run python train_kfold.py --k_folds 5
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
import sys
import argparse
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()

verify_macos_env()

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_clinical_metrics
from train import GPTConfig, GPT, MuonAdamW, norm, build_model_config, device_type, device, autocast_ctx, H100_BF16_PEAK_FLOPS

# Import hyperparameters from train.py
from train import (
    ASPECT_RATIO, HEAD_DIM, WINDOW_PATTERN, TOTAL_BATCH_SIZE,
    EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR,
    WEIGHT_DECAY, ADAM_BETAS, WARMUP_RATIO, WARMDOWN_RATIO,
    FINAL_LR_FRAC, DROPOUT, DEPTH, DEVICE_BATCH_SIZE,
    get_lr_multiplier, get_muon_momentum, get_weight_decay
)

def train_single_fold(fold_idx, k_folds, tokenizer, vocab_size):
    """Train a single fold and return metrics."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{k_folds}")
    print(f"{'='*60}\n")

    # Setup model for this fold
    config = build_model_config(DEPTH)
    print(f"Fold {fold_idx + 1} Model config: {asdict(config)}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print(f"Fold {fold_idx + 1} Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    if device_type == "cuda":
        model = torch.compile(model, dynamic=False)

    # Load fold-specific data
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, f"train_fold{fold_idx}")
    x, y, epoch = next(train_loader)

    print(f"Time budget per fold: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # Training loop for this fold
    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    def sync_device(device_type):
        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

    while True:
        sync_device(device_type)
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)
        MAX_STEPS =130
        progress = min(step / MAX_STEPS, 1.0)
        #progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()

        if train_loss_f > 100:
            print(f"FAIL - Fold {fold_idx + 1}")
            return None

        sync_device(device_type)
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)

        print(f"\rFold {fold_idx+1} - step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | tok/sec: {tok_per_sec:,}    ", end="", flush=True)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        #if step > 10 and total_training_time >= TIME_BUDGET:
        if step > MAX_STEPS:
            break

    print()

    # Evaluate on this fold's validation set
    model.eval()
    with autocast_ctx:
        val_acc, val_auc, val_sens, val_spec,y_true, y_prob= evaluate_clinical_metrics(model, tokenizer, DEVICE_BATCH_SIZE, split_name=f"val_fold{fold_idx}")

    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  val_acc:  {val_acc:.6f}")
    print(f"  val_auc:  {val_auc:.6f}")
    print(f"  val_sens: {val_sens:.6f}")
    print(f"  val_spec: {val_spec:.6f}")
    os.makedirs('saved_models', exist_ok=True)
    model_save_path = f"saved_models/model_fold{fold_idx}.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"  Model saved to {model_save_path}")
    # Cleanup
    del model, optimizer
    gc.collect()
    if device_type == "mps":
        torch.mps.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()

    return {
        'fold': fold_idx + 1,
        'val_acc': val_acc,
        'val_auc': val_auc,
        'val_sens': val_sens,
        'val_spec': val_spec,
        'y_true': y_true,
        'y_prob': y_prob,
    }

def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross Validation for Medical AutoML')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--prepare_data', action='store_true', help='Prepare K-fold data before training')
    args = parser.parse_args()

    k_folds = args.k_folds
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation: K={k_folds}")
    print(f"{'='*60}\n")

    # Prepare K-fold data if requested
    if args.prepare_data:
        print("Preparing K-fold data...")
        import subprocess
        result = subprocess.run(['python', 'prepare_kfold.py', '--k_folds', str(k_folds)],
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error preparing data: {result.stderr}")
            return

    # Load tokenizer
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")
    print()

    # Train each fold
    all_results = []
    t_start_total = time.time()

    for fold_idx in range(k_folds):
        result = train_single_fold(fold_idx, k_folds, tokenizer, vocab_size)
        if result is None:
            print(f"Fold {fold_idx + 1} failed, stopping.")
            return
        all_results.append(result)

    t_end_total = time.time()

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"K-Fold Cross Validation Summary (K={k_folds})")
    print(f"{'='*60}\n")

    print("Individual Fold Results:")
    print(f"{'Fold':<6} {'Accuracy':<10} {'AUC':<10} {'Sensitivity':<12} {'Specificity':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['fold']:<6} {r['val_acc']:<10.4f} {r['val_auc']:<10.4f} {r['val_sens']:<12.4f} {r['val_spec']:<12.4f}")

    # Calculate statistics
    metrics = ['val_acc', 'val_auc', 'val_sens', 'val_spec']
    print("\n" + "="*60)
    print("Summary Statistics (Mean ± Std)")
    print("="*60)

    summary = {}
    for metric in metrics:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary[metric] = {'mean': mean_val, 'std': std_val}

        metric_name = {
            'val_acc': 'Accuracy',
            'val_auc': 'AUC',
            'val_sens': 'Sensitivity',
            'val_spec': 'Specificity'
        }[metric]

        print(f"{metric_name:<15}: {mean_val:.6f} ± {std_val:.6f}")

    print(f"\nTotal time: {(t_end_total - t_start_total)/60:.1f} minutes")
    print(f"Time per fold: {(t_end_total - t_start_total)/k_folds/60:.1f} minutes")

    # Save results
    import json
    results_file = f'results_kfold_{k_folds}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'k_folds': k_folds,
            'individual_results': all_results,
            'summary': summary,
            'total_time_seconds': t_end_total - t_start_total,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print("\n" + "="*60)
    print("K-Fold Cross Validation Complete!")
    print("="*60)

    print("\n---")
    print(f"val_acc_mean:     {summary['val_acc']['mean']:.6f}")
    print(f"val_auc_mean:     {summary['val_auc']['mean']:.6f}")
    print(f"val_sens_mean:    {summary['val_sens']['mean']:.6f}")
    print(f"val_spec_mean:    {summary['val_spec']['mean']:.6f}")
    print("---")

if __name__ == "__main__":
    main()
