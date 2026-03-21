# K-Fold Cross Validation Guide

This guide explains how to use K-fold cross validation to improve model stability and reliability in medical diagnosis tasks.

## Why K-Fold Cross Validation?

### Problem with Single Validation Split
- **High variance**: With only ~30 validation samples (10% of 303), a few misclassifications can drastically change AUC
- **Unstable results**: Running the same experiment multiple times yields different AUC (0.84-0.96)
- **Unreliable comparison**: Hard to determine if improvement is real or just luck

### Benefits of K-Fold
- **More reliable metrics**: Average across K different validation sets
- **Lower variance**: Standard deviation shows consistency
- **Better generalization estimate**: Tests on all data points
- **Publication-ready**: Shows robustness for SCI papers

## Quick Start

### 1. Prepare K-Fold Data (One-time)

```bash
# Create 5-fold cross validation data
uv run python prepare_kfold.py --k_folds 5

# Or use different K value
uv run python prepare_kfold.py --k_folds 10
```

This creates:
- `data/train_fold0.bin` through `data/train_fold4.bin`
- `data/val_fold0.bin` through `data/val_fold4.bin`
- `data/kfold_5_info.txt` (fold distribution info)

### 2. Run K-Fold Training

```bash
# Train all 5 folds (takes ~25 minutes total = 5 folds × 5 minutes)
uv run python train_kfold.py --k_folds 5

# The script automatically prepares data if not exists
uv run python train_kfold.py --k_folds 5 --prepare_data
```

### 3. View Results

The script outputs:
```
Individual Fold Results:
Fold   Accuracy   AUC        Sensitivity  Specificity
------------------------------------------------------------
1      0.7833     0.8931     0.7500       0.8235
2      0.8167     0.9216     0.8000       0.8333
3      0.8000     0.9059     0.7647       0.8571
4      0.8333     0.9412     0.8235       0.8462
5      0.7833     0.8882     0.7353       0.8333

Summary Statistics (Mean ± Std)
============================================================
Accuracy       : 0.803333 ± 0.020408
AUC            : 0.910000 ± 0.020976
Sensitivity    : 0.774706 ± 0.033236
Specificity    : 0.838690 ± 0.012186

Total time: 25.3 minutes
Time per fold: 5.1 minutes

Results saved to: results_kfold_5.json
```

## Understanding the Results

### Individual Fold Results
Shows performance on each validation set. If folds have wildly different performance, the model may be unstable or the data has high variance.

### Summary Statistics
- **Mean**: Average performance across all folds
- **Std (Standard Deviation)**: Measure of consistency
  - Low std (<0.03): Model is stable and reliable
  - High std (>0.05): Model varies significantly across data splits

### Example Interpretation
```
AUC: 0.910000 ± 0.020976
```
- The model achieves **91.0% AUC on average**
- Results are consistent within **±2.1%** across different validation sets
- This is **stable and publication-ready**

## Comparison with Single Split

### Single Split (Before)
```
Run 1: AUC = 0.843
Run 2: AUC = 0.877
Run 3: AUC = 0.961
Range: 0.118 (very unstable!)
```

### K-Fold (After)
```
K-Fold: AUC = 0.910 ± 0.021
Range: 0.042 (much more stable!)
```

## When to Use K-Fold

### ✅ Use K-Fold When:
- Publishing results (SCI papers, conferences)
- Comparing different architectures fairly
- Small dataset (<1000 samples)
- Need to report confidence intervals
- Final model evaluation

### ⚡ Use Single Split When:
- Quick prototyping and debugging
- Hyperparameter exploration (faster iteration)
- Development and testing new features
- Limited compute time

## Best Practices

### 1. Development Workflow
```
Development Phase (Fast Iteration):
  → Use single split
  → Test many hyperparameters quickly
  → Find promising configurations

Final Evaluation Phase (Rigorous):
  → Use K-fold cross validation
  → Report mean ± std
  → Save results for paper
```

### 2. Choosing K Value
- **K=5**: Good balance, 5× training time (recommended)
- **K=10**: More robust, 10× training time (for final evaluation)
- **K=3**: Faster, 3× training time (only if data is very limited)

### 3. Stratification
Current implementation shuffles data randomly. For medical diagnosis:
- Ensure each fold has similar class distribution
- Consider stratified K-fold (preserves class ratios)

## Integration with Autonomous Experiments

You can still use AI agents with K-fold:

1. **Agent searches** using single split (fast)
2. **Once optimal config found**, run K-fold validation
3. **Report final results** with confidence intervals

Example workflow:
```bash
# 1. Agent finds best config using single split (fast iteration)
uv run python train.py  # Run multiple experiments

# 2. Best config: ASPECT_RATIO=48, DROPOUT=0.2, DEPTH=3

# 3. Validate with K-fold (rigorous evaluation)
uv run python train_kfold.py --k_folds 5

# 4. Report: AUC = 0.941 ± 0.021 (mean ± std across 5 folds)
```

## Common Issues

### Issue 1: Out of Memory
**Problem**: Training 5 folds sequentially accumulates memory
**Solution**: Script includes `torch.mps.empty_cache()` between folds

### Issue 2: Long Training Time
**Problem**: 5 folds × 5 min = 25 minutes is slow
**Solution**: 
- Use K=3 for faster validation during development
- Only use K=5 for final results

### Issue 3: Inconsistent Results
**Problem**: High std deviation across folds
**Solution**:
- Check if data is properly shuffled
- Ensure all folds have similar class distribution
- Consider stratified sampling

## Citation in Papers

When reporting K-fold results in your SCI paper:

```
"We evaluated model performance using 5-fold cross validation to ensure 
robustness. The model achieved an AUC of 0.910 ± 0.021 (mean ± std), 
demonstrating consistent performance across different data splits."
```

## Files Generated

After running K-fold:

```
medical-automl/
├── data/
│   ├── train_fold0.bin      # Training data for fold 0
│   ├── val_fold0.bin        # Validation data for fold 0
│   ├── train_fold1.bin
│   ├── val_fold1.bin
│   ├── ... (for all folds)
│   └── kfold_5_info.txt     # Fold distribution information
├── results_kfold_5.json     # Detailed results (JSON format)
└── KFOLD_GUIDE.md           # This guide
```

## Next Steps

1. **Try it now**:
   ```bash
   uv run python prepare_kfold.py --k_folds 5
   uv run python train_kfold.py --k_folds 5
   ```

2. **Analyze results**:
   - Check if std deviation is low (<0.03)
   - Verify all folds have similar performance
   - Use mean ± std for your paper

3. **Compare with single split**:
   - Run single split multiple times
   - Compare variance with K-fold
   - See the stability improvement

## Questions?

- **Why not stratified?**: Current version uses random shuffle. Stratified K-fold (preserving class ratios) can be added if needed.
- **Can I resume interrupted training?**: Not currently supported. Each fold trains independently.
- **How to visualize?**: Results are saved in JSON format. You can load and plot using matplotlib or seaborn.

---

**Recommendation**: Use K-fold for all final evaluations and paper submissions!
