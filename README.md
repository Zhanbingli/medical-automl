# Specificity Collapse and Calibration Drift under External Schema Shift

> **Project status**: public research repository and preprint package in preparation.

## What This Project Asks

How do different clinical prediction pipelines fail when moved off their training distribution?

This repository studies that question in a small clinical prediction setting by comparing:
1. A custom **GPT-style tabular-to-text Transformer** trained from scratch on serialized clinical records
2. A locally deployed **LLM (Qwen3.5-2B)** using 0/1/3/5/10-shot in-context learning
3. **Eight tabular baselines** (Random Forest, XGBoost, Logistic Regression, SVM, Gradient Boosting, MLP, ResNet, TabNet)

The main outcome is not a leaderboard claim. It is an **empirical reliability case study** about how representation choice, schema mismatch, and missing-value encoding affect external behavior.

## What Was Actually Tested

- Internal training/evaluation uses the UCI Heart Disease dataset (`n=303`).
- External evaluation uses the Kaggle Heart Failure Prediction dataset (`n=918`).
- The external dataset does **not** natively contain the UCI `ca` and `thal` variables.
- To run UCI-trained models without retraining, the repo aligns the external schema by introducing `ca=0` and `thal=0` for every external record.
- The custom Transformer consumes a **Chinese serialized template**.
- The LLM consumes an **English natural-language rendering** and sees `thal=0` as `"not available"`.

Because of those choices, this repository should be read as a **stress test / case study**, not as a perfectly controlled architecture comparison.

## Main Findings

| Model / Paradigm | Internal CV (AUC) | External Validation (AUC) | External Behavior |
|---|---|---|---|
| Random Forest (Baseline) | 0.911 ± 0.024 | 0.891 ± 0.042 | Stable |
| Logistic Regression | 0.912 ± 0.018 | — | Strong internal baseline |
| Tabular-to-Text Transformer | 0.762 ± 0.070 | 0.624 ± 0.033 | Specificity drop |
| LLM 0-shot (Qwen3.5-2B) | 0.656 ± 0.058 | 0.597 | Strong positive bias |
| LLM 5-shot (Qwen3.5-2B) | 0.755 ± 0.030 | 0.739 | Best external neural profile |

## Key Limitations

- The external cohort is not a native 13-feature replication of UCI.
- Transformer and LLM do not receive identical surface-form inputs.
- Missingness is encoded differently across the two neural pipelines.
- Baseline preprocessing differs from the Transformer preprocessing in the current repo.
- The study is small and should be treated as hypothesis-generating.

## Preprint Release Files

- Project page: [index.html](index.html)
- OSF metadata: [docs/osf_preprint_metadata.md](docs/osf_preprint_metadata.md)
- OSF upload checklist: [docs/osf_submission_checklist.md](docs/osf_submission_checklist.md)
- Public preprint PDF: [docs/li_2026_clinical_prediction_schema_shift_preprint.pdf](docs/li_2026_clinical_prediction_schema_shift_preprint.pdf)

## Repository Structure

```
├── figures/                           # Publication figures
│   ├── figure1_architecture.*         # Model architecture & pipeline
│   ├── attention_ood_comparison.*     # Attention weight analysis (ID vs OOD)
│   ├── llm_full_metrics.*            # LLM performance across shot regimes
│   ├── llm_generalization_gap.*      # LLM generalization gap analysis
│   ├── llm_shotcurve.*              # Shot-curve learning dynamics
│   ├── sens_spec_tradeoff.*         # Sensitivity-specificity tradeoff
│   ├── dataset_distribution_shift.*  # KDE distribution shift visualization
│   ├── roc_curve_paper_perfect_match.* # ROC curves
│   ├── confusion_matrix_comparison.* # Confusion matrices
│   ├── rf_feature_importance.*       # Random Forest feature importance
│   ├── generalizability_comparison_kaggle.* # External validation comparison
│   └── supplementary/               # Supplementary figures (S1-S9)
├── scripts/                          # Experiment & visualization scripts
│   ├── experiment_ollama.py          # LLM (Qwen3.5-2B) evaluation pipeline
│   ├── experiment_shotcurve.py       # N-shot learning curve experiment
│   ├── experiment_A.py              # Ablation study A (imputation)
│   ├── experiment_B.py              # Ablation study B
│   ├── external_validation.py       # External Kaggle validation
│   ├── statistical_tests_llm.py     # Statistical tests for LLM results
│   ├── plot_figure1_architecture.py # Architecture diagram generation
│   ├── plot_attention_ood_comparison.py # Attention analysis visualization
│   ├── plot_llm_comparison.py       # LLM comparison figures
│   ├── plot_sens_spec_tradeoff.py   # Sensitivity-specificity plots
│   ├── plot_*.py                    # Other visualization scripts
│   └── sci_stats_processor.py       # Statistical processing utilities
├── results/                          # Experimental results (JSON/CSV/TeX)
│   ├── baseline_comparison_5fold.json    # 8-model baseline results
│   ├── experiment_ollama_results.json    # LLM zero-shot & 5-shot results
│   ├── experiment_shotcurve_results.json # 0/1/3/5/10-shot results
│   ├── experiment_A_results.json         # Ablation A results
│   ├── experiment_B_results.json         # Ablation B results
│   ├── external_validation_results.json  # Kaggle external validation
│   ├── results_kfold_5.json             # Transformer 5-fold CV results
│   ├── statistical_tests_results.json   # Baseline statistical tests
│   ├── statistical_tests_llm.json       # LLM statistical tests
│   ├── statistical_tests_llm.tex        # LaTeX table for LLM statistics
│   └── statistical_tests_table.tex      # LaTeX table for baseline stats
├── docs/                             # Documentation
│   ├── KFOLD_GUIDE.md               # K-fold cross validation guide
│   ├── BASELINE_GUIDE.md            # Baseline comparison guide
│   └── STATISTICAL_TESTS_GUIDE.md   # Statistical tests guide
├── prepare.py                        # Data preprocessing & textualization
├── train.py                          # Single-fold Transformer training
├── train_kfold.py                    # 5-fold Transformer training
├── run_baseline_sota.py              # 8-model baseline comparison
├── statistical_tests.py              # Baseline statistical significance
├── evaluate_calibration.py           # Calibration analysis
├── download_data.py                  # Dataset download utility
├── patients.csv                      # UCI Heart Disease dataset (303 records)
├── analysis.ipynb                    # Exploratory data analysis notebook
├── pyproject.toml                    # Project dependencies
└── LICENSE                           # MIT License
```

## Quick Start

### Prerequisites
- Python 3.10+
- Apple Silicon Mac (MPS), NVIDIA GPU, or CPU
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) (for LLM experiments only)

### Installation

```bash
git clone https://github.com/Zhanbingli/medical-automl.git
cd medical-automl

# Install dependencies
uv sync
```

### Reproduce Experiments

```bash
# 1. Prepare data and train tokenizer (~2 min)
uv run python prepare.py

# 2. Train Transformer with 5-fold CV (~25 min)
uv run python train_kfold.py --k_folds 5

# 3. Run baseline comparison (~25-30 min)
uv run python run_baseline_sota.py

# 4. External validation on Kaggle cohort
uv run python scripts/external_validation.py

# 5. LLM evaluation (requires Ollama with qwen3.5:2b)
uv run python scripts/experiment_ollama.py

# 6. Shot-curve learning dynamics
uv run python scripts/experiment_shotcurve.py

# 7. Statistical significance testing
uv run python statistical_tests.py
uv run python scripts/statistical_tests_llm.py
```

### Generate Figures

```bash
# All figure scripts output to figures/
uv run python scripts/plot_figure1_architecture.py
uv run python scripts/plot_attention_ood_comparison.py
uv run python scripts/plot_llm_comparison.py
uv run python scripts/plot_sens_spec_tradeoff.py
uv run python scripts/plot_dataset_shift.py
uv run python scripts/plot_roc.py
uv run python scripts/plot_confusion_matrix.py
uv run python scripts/plot_generaliza.py
```

## Methods Summary

### Tabular-to-Text Pipeline
Structured patient records → fixed Chinese serialized template → BPE tokenization (vocab=8,192) → GPT-style Transformer with rotary positional embeddings.

### LLM In-Context Learning
Patient records → English narrative → Qwen3.5-2B via Ollama → JSON probability output → threshold at 0.5. N-shot examples sampled with balanced positive/negative cases (seed=42).

### Evaluation Protocol
- **Internal**: 5-fold stratified CV (StratifiedKFold, seed=42) — all models use identical splits
- **External**: Aligned Kaggle cohort (n=918) where missing UCI-only fields `ca`/`thal` are introduced as schema-alignment variables and set to `0`
- **Statistics**: Paired t-tests with Bonferroni correction, Cohen's d effect sizes, 95% CIs

## Dataset

| Cohort | Source | n | Features | Imputation |
|---|---|---|---|---|
| Internal (training) | UCI Heart Disease (Cleveland) | 303 | 13 clinical features | Missing → 0 (silent) |
| External (validation) | Kaggle Heart Failure Prediction | 918 | 11 native features + 2 aligned fields | `ca=0`, `thal=0` introduced to match UCI schema |

The critical external stressor is not a naturally matched 13-feature replication, but an aligned schema shift: the external dataset lacks `ca` and `thal`, so those fields are synthetically introduced for compatibility with UCI-trained models. This makes `thal=0` an especially important out-of-distribution token for the Transformer pipeline.

## Citation

```bibtex
@misc{li2026specificity,
  author  = {Li, Zhanbing},
  title   = {Specificity Collapse and Calibration Drift under External Schema Shift: An Empirical Case Study of Tabular-to-Text Transformers and Large Language Models in Clinical Prediction},
  year    = {2026},
  note    = {Preprint in preparation},
  url     = {https://github.com/Zhanbingli/medical-automl}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Zhanbing Li
- **Email**: zhanbing2025@gmail.com
- **ORCID**: 0009-0003-6067-2183
- **GitHub Issues**: [https://github.com/Zhanbingli/medical-automl/issues](https://github.com/Zhanbingli/medical-automl/issues)

---

**Disclaimer**: This project is for research and educational purposes only. Not intended for clinical use without proper validation and regulatory approval.
