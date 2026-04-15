# Workflow

Use this order unless the user asks for a narrower scope.

## Preflight

1. Run `uv sync`.
2. Confirm whether the request needs strict reproduction or only an audit.
3. Check platform and service prerequisites from `known_gaps.md`.
4. Inspect existing result files before rerunning expensive jobs.

## Core Transformer Path

1. `uv run python prepare.py`
2. `uv run python train_kfold.py --k_folds 5`
3. Verify `data/tokenizer.pkl`, `data/train.bin`, `data/val.bin`, `saved_models/model_fold0.pt` through `saved_models/model_fold4.pt`, and `results_kfold_5.json`

This path is blocked on non-macOS systems or when MPS is unavailable because both `prepare.py` and `train_kfold.py` enforce that gate.

## Baseline Path

1. `uv run python run_baseline_sota.py`
2. Verify `baseline_comparison_5fold.json`
3. If needed, verify `rf_feature_importance.png` and `rf_feature_importance.pdf`

## External Validation Path

1. Ensure the Kaggle CSV exists at the cache path used by `scripts/external_validation.py`
2. If it is missing, run `uv run python download_data.py`
3. Run `uv run python scripts/external_validation.py`
4. Verify `external_validation_results.json`

This path also requires the saved fold models from the core transformer path.

## Statistics Path

1. Run `uv run python statistical_tests.py`
2. Verify `statistical_tests_results.json` and `statistical_tests_table.tex`

For LLM statistics:

1. Ensure the LLM result JSON files exist under `results/`
2. Run `uv run python scripts/statistical_tests_llm.py`
3. Verify `results/statistical_tests_llm.json` and `results/statistical_tests_llm.tex`

## LLM Path

1. Confirm `ollama serve` is reachable at `http://localhost:11434`
2. Confirm model `qwen3.5:2b` is installed
3. Ensure the Kaggle CSV exists if external LLM validation is requested
4. Run `uv run python scripts/experiment_ollama.py`
5. Run `uv run python scripts/experiment_shotcurve.py` only if the user needs 1-shot, 3-shot, and 10-shot results
6. Verify `results/experiment_ollama_results.json` and optionally `results/experiment_shotcurve_results.json`

## Figure Path

Run only the figure scripts that match available upstream artifacts.

Common dependencies:

- `evaluate_calibration.py` needs `results_kfold_5.json`
- `scripts/generate_paper_figures.py` needs `results_kfold_5.json`, `external_validation_results.json`, and `statistical_tests_results.json`
- several `scripts/plot_*.py` readers expect JSON files under `results/`

Do not run broad figure generation if the needed upstream JSON files are missing or stale.
