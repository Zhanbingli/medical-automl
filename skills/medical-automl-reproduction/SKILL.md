---
name: medical-automl-reproduction
description: Use when reproducing, validating, or auditing experiments in this medical-automl repository, including environment setup, data preparation, transformer k-fold training, baseline comparison, external validation, Ollama LLM experiments, statistics, and figure generation.
---

# Medical AutoML Reproduction

Use this skill only for this repository.

This repo is reproducible only under explicit constraints. Do not claim full reproduction until you have checked the environment gates and verified the expected artifacts.

## First Reads

Read these files first:

- `pyproject.toml`
- `README.md`
- `references/workflow.md`
- `references/known_gaps.md`

Then read `references/commands.md` and `references/artifacts.md` only for the parts needed by the user's request.

## Scope Routing

Classify the request into one or more of these scopes:

1. Core transformer reproduction
2. Baseline comparison
3. External validation
4. LLM reproduction with Ollama
5. Statistical analysis
6. Figure generation
7. Reproducibility audit only

## Environment Gates

Check these before running expensive steps:

- `uv` environment available and dependencies install with `uv sync`
- `prepare.py` and `train_kfold.py` require macOS plus `torch.backends.mps.is_available()`
- external validation requires the Kaggle Heart Failure CSV in the cache path used by the scripts, or a successful `download_data.py` run
- LLM experiments require a running Ollama server plus the `qwen3.5:2b` model

If any gate fails, switch to constrained reproduction: run the compatible subset, report the blocker, and state exactly which steps were not executed.

## Execution Protocol

1. Prefer existing artifacts before rerunning long jobs.
2. Run the minimum scope needed for the user request.
3. Follow the command order in `references/workflow.md`.
4. After every step, verify the producer's expected files from `references/artifacts.md`.
5. Never use `train_kfold.py --prepare_data` unless `prepare_kfold.py` exists; this repo currently references that file but does not ship it.
6. Treat root-level JSON outputs and `results/` JSON outputs as different locations. Some producers write to the repo root while some consumers read from `results/`.
7. If downstream scripts require a mirrored JSON file under `results/`, create a non-destructive copy instead of moving the original, and say why.
8. Never claim that figures or statistics are fresh unless the upstream result files were freshly verified in this run or the user explicitly accepts existing artifacts.

## Success Standard

Strict reproduction means:

- prerequisites were satisfied
- commands were actually executed
- expected artifacts were produced or refreshed
- the final report names the exact files used

Constrained reproduction means:

- only the compatible subset ran
- missing prerequisites are listed
- no unrun step is described as completed

## Reporting

When closing the task, report:

- which scopes were attempted
- which commands actually ran
- which artifacts were produced or reused
- which blockers remain
- whether the result is strict reproduction or constrained reproduction
