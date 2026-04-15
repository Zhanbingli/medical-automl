# Known Gaps

This file lists repository-specific constraints that an agent must treat as real blockers, not optional details.

## Hard Platform Gate

`prepare.py` and `train_kfold.py` call `verify_macos_env()` at import time.

That means:

- non-macOS systems cannot run them
- macOS systems without MPS cannot run them

Do not claim full reproduction if this gate fails.

## Missing File Reference

`train_kfold.py --prepare_data` tries to call `prepare_kfold.py`, but that file is not present in this repository snapshot.

Do not use `--prepare_data` unless the file has been added.

## Kaggle Path Assumption

`scripts/external_validation.py` and `scripts/experiment_ollama.py` expect the Kaggle Heart Failure dataset in the cache path used by `kagglehub`.

Default expected file:

- `~/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv`

`download_data.py` can populate that cache, but network access and Kaggle availability are outside the repo.

## Ollama Requirement

The LLM path is not reproducible unless:

- Ollama is running locally
- the REST API is reachable at `http://localhost:11434`
- model `qwen3.5:2b` is installed

## Artifact Path Drift

There is a mismatch between some producers and consumers:

- several core scripts write root-level JSON files
- several LLM and plotting scripts read from `results/`

An agent must verify real file locations before chaining steps.

## Stale Documentation Risk

Some docs mention files or paths that do not exactly match current code.

When docs and code disagree, trust the code paths and argument parsers.
