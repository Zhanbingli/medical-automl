# Highlights

- Tabular-to-text Transformers exhibit Specificity Collapse on external validation: a single out-of-distribution categorical token reduces attention to a critical clinical feature by 71.5%, causing false-positive bias.
- LLM few-shot clinical prediction follows an inverted-U pattern: optimal at 5-shot, collapsing at 10-shot to match Transformer degradation magnitude, consistent with in-context overfitting.
- Mean imputation paradoxically worsens tabular-to-text model performance (ΔAUC = −0.125) because tokenization-level familiarity outweighs statistical accuracy.
- Transformers and LLMs exhibit mechanistically distinct generalization failure modes — attention-routing disruption versus calibration drift — requiring different deployment mitigation strategies.
- The 5-shot LLM is the only neural model achieving deployment-stable specificity (>0.60) with AUC >0.73 on external validation without retraining.
