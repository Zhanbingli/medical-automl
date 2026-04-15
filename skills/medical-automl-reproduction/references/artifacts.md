# Artifacts

Verify producer outputs after each stage.

## Core Transformer

Producer: `prepare.py`

- `data/tokenizer.pkl`
- `data/train.bin`
- `data/val.bin`

Producer: `train_kfold.py`

- `results_kfold_5.json`
- `saved_models/model_fold0.pt`
- `saved_models/model_fold1.pt`
- `saved_models/model_fold2.pt`
- `saved_models/model_fold3.pt`
- `saved_models/model_fold4.pt`

## Baselines

Producer: `run_baseline_sota.py`

- `baseline_comparison_5fold.json`
- `rf_feature_importance.png`
- `rf_feature_importance.pdf`

## External Validation

Producer: `scripts/external_validation.py`

- `external_validation_results.json`

## Statistics

Producer: `statistical_tests.py`

- `statistical_tests_results.json`
- `statistical_tests_table.tex`

Producer: `scripts/statistical_tests_llm.py`

- `results/statistical_tests_llm.json`
- `results/statistical_tests_llm.tex`

## LLM

Producer: `scripts/experiment_ollama.py`

- `results/experiment_ollama_results.json`
- `results/experiment_ollama_checkpoint.json`

Producer: `scripts/experiment_shotcurve.py`

- `results/experiment_shotcurve_results.json`
- `results/experiment_shotcurve_checkpoint.json`

## Calibration And Figures

Producer: `evaluate_calibration.py`

- `calibration_analysis_perfect_match.png`
- `calibration_analysis_perfect_match.pdf`

Producer: `scripts/generate_paper_figures.py`

- `figure3_combined_analysis.png`
- `figure3_combined_analysis.pdf`

## Location Mismatch Rule

This repo is inconsistent about JSON output locations.

- `train_kfold.py`, `run_baseline_sota.py`, `statistical_tests.py`, `scripts/external_validation.py`, `evaluate_calibration.py`, and `scripts/generate_paper_figures.py` write or read root-level files
- `scripts/experiment_ollama.py`, `scripts/experiment_shotcurve.py`, and several plotting or LLM statistics scripts read from `results/`

Before a downstream step, verify the exact path that its reader expects. If needed, create a copy under `results/` and say that you normalized the artifact layout for compatibility.
