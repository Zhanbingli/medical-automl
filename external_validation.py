"""
Medical AutoML - External Validation on Kaggle Heart Failure Dataset

Pipeline:
  1. Convert Kaggle CSV → Chinese text (same format as prepare.py)
  2. Encode with existing BPE tokenizer (no retraining)
  3. Run inference with each saved fold model
  4. Aggregate results across 5 folds

Usage: uv run python external_validation.py

Requirements:
  - data/tokenizer.pkl  (trained by prepare.py)
  - saved_models/model_fold{0..4}.pt  (trained by train_kfold.py)
  - Kaggle dataset at the path below
"""

import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# ── path config ───────────────────────────────────────────────────────────────
KAGGLE_CSV    = '/Users/lizhanbing12/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv'
TOKENIZER_PKL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tokenizer.pkl")
MODELS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
OUTPUT_JSON   = "external_validation_results.json"
K_FOLDS       = 5

# ── import from your existing code ───────────────────────────────────────────
from prepare import MAX_SEQ_LEN, Tokenizer, BOS_TOKEN
from train import GPT, build_model_config, DEPTH

print("=" * 70)
print("External Validation: UCI (train) → Kaggle Heart Failure (n=918)")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Kaggle → UCI-compatible Chinese text
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading and converting Kaggle dataset...")

import pandas as pd
df = pd.read_csv(KAGGLE_CSV)

SEX_MAP = {'M': 1, 'F': 0}
CP_MAP  = {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4}
ECG_MAP = {'Normal': 0, 'ST': 1, 'LVH': 2}
ANG_MAP = {'Y': 1, 'N': 0}
SLP_MAP = {'Up': 1, 'Flat': 2, 'Down': 3}

texts  = []
labels = []

for _, row in df.iterrows():
    r = {
        'age'     : int(row['Age']),
        'sex'     : SEX_MAP.get(str(row['Sex']), 0),
        'cp'      : CP_MAP.get(str(row['ChestPainType']), 0),
        'trestbps': int(row['RestingBP']),
        'chol'    : int(row['Cholesterol']),
        'fbs'     : int(row['FastingBS']),
        'restecg' : ECG_MAP.get(str(row['RestingECG']), 0),
        'thalach' : int(row['MaxHR']),
        'exang'   : ANG_MAP.get(str(row['ExerciseAngina']), 0),
        'oldpeak' : float(row['Oldpeak']),
        'slope'   : SLP_MAP.get(str(row['ST_Slope']), 0),
        'ca'      : 0,   # not present in Kaggle → zero-pad
        'thal'    : 0,   # not present in Kaggle → zero-pad
        'label'   : int(row['HeartDisease']),
    }
    # MUST match prepare.py load_and_process_patients() exactly
    text = (
        f"患者特征：年龄{r['age']}，性别{r['sex']}，"
        f"胸痛类型{r['cp']}，静息血压{r['trestbps']}，"
        f"胆固醇{r['chol']}，空腹血糖{r['fbs']}，"
        f"静息心电图{r['restecg']}，最大心率{r['thalach']}，"
        f"运动诱发心绞痛{r['exang']}，ST段压低{r['oldpeak']}，"
        f"ST段斜率{r['slope']}，透视血管数{r['ca']}，"
        f"地中海贫血{r['thal']}。最终诊断结果为：{r['label']}"
    )
    texts.append(text)
    labels.append(r['label'])

labels = np.array(labels)
print(f"  Converted {len(texts)} records to Chinese text")
print(f"  Class dist: {{0: {(labels==0).sum()}, 1: {(labels==1).sum()}}}")
print(f"  Sample: {texts[0][:80]}...")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Encode with existing BPE tokenizer
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Encoding with existing BPE tokenizer (no retraining)...")

assert os.path.exists(TOKENIZER_PKL), \
    f"Tokenizer not found: {TOKENIZER_PKL}\nPlease run prepare.py first."

with open(TOKENIZER_PKL, 'rb') as f:
    enc = pickle.load(f)

tokenizer = Tokenizer(enc)
bos_id    = tokenizer.get_bos_token_id()

try:
    token_id_1 = enc.encode_single_token('1')
except Exception:
    token_id_1 = enc.encode_ordinary('1')[-1]

print(f"  Vocab size   : {tokenizer.get_vocab_size():,}")
print(f"  BOS token id : {bos_id}  |  '1' token id : {token_id_1}")

# context = all tokens except the final label token (which we predict)
encoded_records = []
for text, label in zip(texts, labels):
    ids = enc.encode_ordinary(text)
    ids.insert(0, bos_id)
    context = ids[:-1]
    if len(context) > MAX_SEQ_LEN:
        context = context[-MAX_SEQ_LEN:]
    encoded_records.append((context, int(label)))

avg_len = np.mean([len(c) for c, _ in encoded_records])
print(f"  Encoded {len(encoded_records)} records  (avg {avg_len:.1f} tokens)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Inference with each saved fold model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Running inference with saved fold models...")

device_str = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
device = torch.device(device_str)
print(f"  Device: {device}")

fold_results = []

for fold_idx in range(K_FOLDS):
    model_path = os.path.join(MODELS_DIR, f"model_fold{fold_idx}.pt")
    if not os.path.exists(model_path):
        print(f"  [!] model_fold{fold_idx}.pt not found, skipping.")
        continue

    print(f"\n  ── Fold {fold_idx} ──────────────────────────")

    config = build_model_config(DEPTH)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    y_true, y_prob, y_pred = [], [], []

    with torch.no_grad():
        for context, label in encoded_records:
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits_all = model(x)           # (1, seq_len, vocab)
            last_logits = logits_all[0, -1] # (vocab,)
            probs  = F.softmax(last_logits, dim=-1)
            prob_1 = probs[token_id_1].item()
            pred   = 1 if last_logits.argmax().item() == token_id_1 else 0
            y_true.append(label)
            y_prob.append(prob_1)
            y_pred.append(pred)

    auc  = roc_auc_score(y_true, y_prob)
    acc  = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  AUC={auc:.4f}  Acc={acc:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")
    fold_results.append({'fold': fold_idx, 'auc': auc, 'acc': acc,
                         'sensitivity': sens, 'specificity': spec})

    del model
    if device_str == 'mps':   torch.mps.empty_cache()
    elif device_str == 'cuda': torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregate & report
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Aggregating results...")

if not fold_results:
    print("No results. Ensure saved_models/model_fold*.pt exist.")
    sys.exit(1)

summary = {}
for m in ['auc', 'acc', 'sensitivity', 'specificity']:
    vals = [r[m] for r in fold_results]
    summary[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION RESULTS")
print(f"  Training cohort : UCI Cardiovascular Disease (n=297)")
print(f"  External cohort : Kaggle Heart Failure Prediction (n={len(texts)})")
print(f"  Models used     : {len(fold_results)} saved fold models (no retraining)")
print("=" * 70)
for m, label in [('auc','AUC'), ('acc','Accuracy'),
                  ('sensitivity','Sensitivity'), ('specificity','Specificity')]:
    print(f"  {label:<12}: {summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}")
print("=" * 70)
print("  Note: 'ca' and 'thal' absent in Kaggle dataset → zero-padded.")

# save JSON
output = {
    'external_cohort': f'Kaggle Heart Failure Prediction (n={len(texts)})',
    'training_cohort': 'UCI Cardiovascular Disease (n=297)',
    'n_fold_models'  : len(fold_results),
    'fold_results'   : fold_results,
    'summary'        : summary,
    'feature_note'   : 'ca and thal not available in Kaggle dataset, zero-padded to 0',
}
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved to: {OUTPUT_JSON}")

# LaTeX snippet
am, as_ = summary['auc']['mean'], summary['auc']['std']
print(f"""
LaTeX snippet:
--------------
The agent-optimized Transformer, trained exclusively on the UCI cohort
($n$=297), was evaluated on an independent external dataset of
{len(texts)} patients without any retraining or fine-tuning.
Inference was performed using all {len(fold_results)} fold models and
results averaged to reduce variance.
The model achieved a mean AUC of ${am:.3f} \\pm {as_:.3f}$,
a sensitivity of {summary['sensitivity']['mean']:.3f}, and a specificity
of {summary['specificity']['mean']:.3f}, demonstrating cross-dataset
generalizability of the agent-discovered architecture.
(Limitation: \\texttt{{ca}} and \\texttt{{thal}} features absent in the
external cohort were zero-padded.)
""")
