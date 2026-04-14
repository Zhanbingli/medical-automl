# OSF Preprint Metadata

Use this file as a copy-paste source when creating the OSF preprint record.

## Title

Specificity Collapse and Calibration Drift under External Schema Shift: An Empirical Case Study of Tabular-to-Text Transformers and Large Language Models in Clinical Prediction

## Abstract

Clinical prediction models often degrade under external deployment, but the resulting failure patterns may depend on representation and inference paradigm. We studied this question in a small-scale clinical prediction setting using a tabular-to-text Transformer, a frozen large language model (LLM), and conventional tabular baselines.

We trained a custom GPT-style tabular-to-text Transformer on the UCI Heart Disease dataset (n=303; 5-fold stratified cross-validation) and compared it with eight ML baselines and a locally deployed LLM (Qwen3.5-2B) under 0-, 1-, 3-, 5-, and 10-shot prompting. External evaluation used a 918-patient Kaggle heart disease dataset with only 11 corresponding predictors; two UCI features (`ca`, `thal`) were introduced as alignment variables and set to 0 to match the training schema. The Transformer used a Chinese tabular-to-text pipeline, whereas the LLM used an English natural-language rendering with semantically explicit missing-value wording. Primary metrics were AUC, sensitivity, and specificity.

The Transformer achieved internal AUC 0.762 ± 0.070 and specificity 0.811 ± 0.084, but on the aligned external cohort its mean AUC fell to 0.624 ± 0.033 and specificity to 0.656 ± 0.279. Attention comparisons on matched inputs were consistent with disrupted routing around the `thal` token, but should be interpreted as mechanistic probes rather than definitive causal proof. The LLM showed strong zero-shot positive bias, whereas 5-shot prompting yielded the most stable external neural profile. In an ablation that altered the aligned external encoding, mean imputation of `ca`/`thal` degraded Transformer performance relative to zero-padding, suggesting sensitivity to tokenization-level familiarity.

This preprint should be read as a clinical AI reliability case study rather than a definitive architecture ranking. In this setup, the tabular-to-text Transformer and the prompted LLM exhibited different external failure patterns, but the comparison also includes differences in representation language, missing-value semantics, and external schema alignment.

## Keywords

- clinical AI reliability
- clinical prediction
- distribution shift
- external validation
- in-context learning
- large language models
- schema shift
- tabular-to-text

## Suggested Subjects

- Computer Science
- Machine Learning
- Health Informatics
- Artificial Intelligence
- Medicine and Health Sciences

Pick the OSF category labels that most closely match these themes.

## Authors

- Zhanbing Li
- ORCID: 0009-0003-6067-2183
- Affiliation 1: School of Clinical Medicine, Kunming Medical University, Kunming, China
- Affiliation 2: Department of Cardiology, Yuxi People's Hospital, Yuxi, Yunnan, China
- Email: zhanbing2025@gmail.com

## Links

- Code repository: https://github.com/Zhanbingli/medical-automl
- Project figures/results: include the GitHub repository link above unless you create a separate project page

## Suggested License

Recommended default: CC BY 4.0

Conservative alternative: CC BY-NC-ND 4.0

Choose one and use the same license wording consistently on OSF and in the manuscript footer if you add one.

## Data and Code Availability Statement

Code, experimental scripts, generated figures, and processed results are available in the public GitHub repository linked above. The study uses publicly accessible benchmark datasets. External evaluation requires the Kaggle Heart Failure Prediction dataset, whose access and license terms are governed by its original source.

## Competing Interests

The author declares no competing interests.

## Funding

No external funding was received for this work.

## Ethics / Human Subjects

This study used only publicly available secondary datasets and did not involve new patient recruitment or intervention.

## Suggested Citation Block

Li Z. Specificity Collapse and Calibration Drift under External Schema Shift: An Empirical Case Study of Tabular-to-Text Transformers and Large Language Models in Clinical Prediction. Preprint. 2026. Available from: [OSF URL after upload].
