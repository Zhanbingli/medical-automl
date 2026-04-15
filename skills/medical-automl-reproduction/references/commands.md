# Commands

Use these commands as the default entry points.

## Environment

```bash
uv sync
```

## Core Transformer Reproduction

```bash
uv run python prepare.py
uv run python train_kfold.py --k_folds 5
```

## Baselines

```bash
uv run python run_baseline_sota.py
```

## External Validation

```bash
uv run python download_data.py
uv run python scripts/external_validation.py
```

## Statistics

```bash
uv run python statistical_tests.py
uv run python scripts/statistical_tests_llm.py
```

## LLM Experiments

```bash
uv run python scripts/experiment_ollama.py
uv run python scripts/experiment_shotcurve.py
```

## Figures

```bash
uv run python evaluate_calibration.py
uv run python scripts/generate_paper_figures.py
uv run python scripts/plot_figure1_architecture.py
uv run python scripts/plot_attention_ood_comparison.py
uv run python scripts/plot_llm_comparison.py
uv run python scripts/plot_sens_spec_tradeoff.py
uv run python scripts/plot_dataset_shift.py
uv run python scripts/plot_roc.py
uv run python scripts/plot_confusion_matrix.py
uv run python scripts/plot_generaliza.py
```

## Preflight Checks

Platform gate:

```bash
uv run python -c "import sys, torch; print(sys.platform); print(torch.backends.mps.is_available())"
```

Ollama gate:

```bash
curl -s http://localhost:11434/api/tags
```

Kaggle cache gate:

```bash
test -f ~/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv
```
