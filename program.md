# Medical AutoML - Clinical Research Protocol

This document guides AI agents through autonomous experimentation for cardiovascular disease diagnosis optimization.

## Project Overview

**Goal**: Develop an optimal transformer-based diagnostic model for cardiovascular disease using automated architecture search.

**Primary Metric**: `val_auc` (Area Under ROC Curve) - prioritizes clinical utility over raw accuracy.

**Secondary Metrics**:
- `val_acc` - Overall accuracy
- `val_sens` - Sensitivity (minimize false negatives)
- `val_spec` - Specificity (minimize false positives)

## Experiment Setup

### 1. Branch Creation
Create a new branch for each experimental series:
```bash
git checkout -b experiments/<date>-<focus>
# Example: experiments/mar11-width-optimization
```

### 2. Initial Setup Checklist
- [ ] Read current `train.py` to understand the architecture
- [ ] Verify data exists: `data/train.bin`, `data/val.bin`, `data/tokenizer.pkl`
- [ ] Initialize results file: `results_clinical.tsv` with headers
- [ ] Confirm current baseline metrics

### 3. Data Files
- **Source**: `patients.csv` (303 cardiovascular patient records)
- **Processed**: `data/train.bin`, `data/val.bin` (binary token sequences)
- **Tokenizer**: `data/tokenizer.pkl` (BPE, 8K vocab)

## Experimentation Guidelines

### What You CAN Modify
Edit `train.py` freely:
- Model architecture (depth, width, attention patterns)
- Hyperparameters (learning rates, dropout, batch size)
- Optimizer settings
- Training procedures

### Constraints
- **Time Budget**: 5 minutes wall-clock training
- **Evaluation**: Fixed clinical metrics in `prepare.py`
- **Dependencies**: Only use packages in `pyproject.toml`

### Success Criteria
1. **Primary**: Maximize `val_auc`
2. **Clinical Balance**: Maintain reasonable sensitivity/specificity trade-off
3. **Stability**: Results should be reproducible

## Workflow

### Phase 1: Baseline
Always start by running the current configuration:
```bash
python train.py > run.log 2>&1
```

Extract results:
```bash
grep "^val_auc:\|^val_acc:\|^val_sens:\|^val_spec:" run.log
```

### Phase 2: Iterative Improvement

**LOOP**:
1. Propose a hypothesis (e.g., "increasing width improves AUC")
2. Modify `train.py` with specific changes
3. Commit: `git commit -m "Exp: <description>"`
4. Run: `python train.py > run.log 2>&1`
5. Extract metrics and compare

**Decision**:
- If `val_auc` improves → Keep commit, continue
- If `val_auc` decreases or equal → `git reset --hard HEAD~1`

### Phase 3: Logging

Record all experiments in `results_clinical.tsv`:

```
commit	val_auc	val_acc	memory_gb	status	description
abc1234	0.941176	0.827586	0.0	keep	baseline
xyz5678	0.950000	0.850000	0.0	keep	increased learning rate
```

Columns:
1. `commit`: Short git hash (7 chars)
2. `val_auc`: Primary metric
3. `val_acc`: Secondary metric
4. `memory_gb`: Peak VRAM in GB
5. `status`: `keep`, `discard`, or `crash`
6. `description`: Brief experimental change

## Architecture Design Space

### Model Dimensions
- **DEPTH**: Number of transformer layers (typically 3-8)
- **ASPECT_RATIO**: Controls model width (typically 32-64)
- **HEAD_DIM**: Attention head dimension (typically 64-128)

### Key Hyperparameters
```python
ASPECT_RATIO = 48      # Model width factor
HEAD_DIM = 128         # Per-head dimension
DROPOUT = 0.2          # Regularization
DEPTH = 3              # Layer count

# Learning rates
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.002
MATRIX_LR = 0.025
SCALAR_LR = 0.5
```

### Optimization Strategies
- Muon optimizer for 2D parameters
- AdamW for embeddings and scalars
- Learning rate warmup/warmdown
- Weight decay scheduling

## Clinical Considerations

### Why AUC over Accuracy?
In medical diagnosis:
- **Sensitivity** (recall): Missing sick patients is costly
- **Specificity**: False alarms waste resources
- **AUC**: Balances both across all thresholds

### Target Performance
- **AUC > 0.90**: Excellent discrimination
- **Sensitivity > 0.80**: Catch most true cases
- **Specificity > 0.80**: Avoid excessive false positives

## Best Practices

### Hypothesis Formation
- Start with architectural changes (depth, width)
- Then tune regularization (dropout)
- Finally optimize learning rates

### Change Magnitude
- Make small, isolated changes
- One variable at a time when possible
- Document expected vs. actual outcomes

### Failure Recovery
If a run crashes:
1. Check `run.log` for stack traces
2. Simple typos → fix and retry
3. Fundamental flaws → discard and revert

## Current Best Configuration

```python
ASPECT_RATIO = 48
DROPOUT = 0.2
DEPTH = 3

# Achieved: val_auc = 0.941176
#           val_acc = 0.827586
#           val_sens = 0.823529
#           val_spec = 1.000000
```

Use this as your optimization starting point.

## Advanced Techniques

### Architecture Patterns
- **Window Patterns**: Try "SSSL", "SL", "L" for attention
- **GQA**: Grouped Query Attention for efficiency
- **VE (Value Embeddings)**: Alternating layer enhancement

### Hyperparameter Search
- Grid search learning rates in [0.001, 0.1]
- Bayesian optimization for architecture
- Population-based training

### Diagnostic Tools
Check model behavior:
- Confusion matrix breakdown
- ROC curve analysis
- Calibration metrics

## Notes

- **Platform**: Optimized for Apple Silicon (MPS) and NVIDIA GPUs
- **Time Budget**: Fixed at 300 seconds regardless of hardware
- **Reproducibility**: Set `torch.manual_seed(42)` for consistency

## References

- Transformer architecture: Vaswani et al. (2017)
- Muon optimizer: Distributed Shampoo variants
- Clinical metrics: Sensitivity, Specificity, AUC interpretation

---

**Remember**: In medical AI, reliability and interpretability matter as much as performance. Aim for consistent, reproducible improvements.
