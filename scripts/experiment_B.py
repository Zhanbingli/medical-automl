"""
实验 B：英文 vs 中文叙述编码 对比实验
=====================================
验证编码语言对模型性能的影响

步骤：
  1. 用英文模板重新生成数据并训练 BPE tokenizer
  2. 跑 5-fold CV，得到英文编码的内部验证 AUC
  3. 与中文编码的 AUC=0.762±0.070 对比

运行方式：
    python experiment_B_language.py

注意：本实验需要重新训练，约 30 分钟（与原实验相同）
"""

import os
import sys
import json
import pickle
import time
import gc
import argparse
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

# ── 环境检查 ────────────────────────────────────────────────
if sys.platform != "darwin":
    raise RuntimeError("This script requires macOS with MPS.")
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available.")

# ── 复用原项目的基础设施 ──────────────────────────────────
from prepare import (MAX_SEQ_LEN, TIME_BUDGET, BOS_TOKEN, SPECIAL_TOKENS,
                     SPLIT_PATTERN, VOCAB_SIZE, CSV_PATH)
from train import (GPTConfig, GPT, build_model_config, device, device_type,
                   autocast_ctx, DEPTH, ASPECT_RATIO, HEAD_DIM,
                   TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE,
                   EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR,
                   WEIGHT_DECAY, ADAM_BETAS,
                   get_lr_multiplier, get_muon_momentum, get_weight_decay)

# ── 英文实验专用数据目录 ──────────────────────────────────
EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_english")
os.makedirs(EXP_DIR, exist_ok=True)

K_FOLDS   = 5
MAX_STEPS = 130   # 与原实验一致

# ────────────────────────────────────────────────────────────
#  英文叙述模板（与中文模板结构完全对应）
# ────────────────────────────────────────────────────────────

def row_to_english(row):
    """将一行患者数据转为英文叙述（与中文模板字段一一对应）"""
    sex_str    = "male"   if row['sex']   == 1 else "female"
    exang_str  = "yes"    if row['exang'] == 1 else "no"
    fbs_str    = "yes"    if row['fbs']   == 1 else "no"

    cp_map     = {1: "typical angina", 2: "atypical angina",
                  3: "non-anginal pain", 4: "asymptomatic"}
    restecg_map= {0: "normal", 1: "ST-T abnormality", 2: "LVH"}
    slope_map  = {1: "upsloping", 2: "flat", 3: "downsloping"}

    cp_str      = cp_map.get(int(row['cp']),      str(row['cp']))
    restecg_str = restecg_map.get(int(row['restecg']), str(row['restecg']))
    slope_str   = slope_map.get(int(row['slope']),    str(row['slope']))

    label = int(row['num_binary'])

    text = (
        f"Patient features: age {row['age']}, sex {sex_str}, "
        f"chest pain type {cp_str}, resting blood pressure {row['trestbps']}, "
        f"cholesterol {row['chol']}, fasting blood sugar above 120 {fbs_str}, "
        f"resting ECG {restecg_str}, maximum heart rate {row['thalach']}, "
        f"exercise-induced angina {exang_str}, ST depression {row['oldpeak']}, "
        f"ST slope {slope_str}, major vessels {row['ca']}, "
        f"thalassemia {row['thal']}. Final diagnosis: {label}"
    )
    return text, label


def prepare_english_data():
    """生成英文数据、训练 tokenizer、保存 K-fold bin 文件"""
    print("── 步骤 1：生成英文叙述数据 ──────────────────────")
    df = pd.read_csv(CSV_PATH)
    df['num_binary'] = (df['num'] > 0).astype(int)
    df = df.replace('?', float('nan'))

    texts, labels = [], []
    for _, row in df.iterrows():
        t, l = row_to_english(row)
        texts.append(t)
        labels.append(l)

    print(f"  生成 {len(texts)} 条英文叙述")

    # ── 训练英文 BPE tokenizer ─────────────────────────
    tok_path = os.path.join(EXP_DIR, "tokenizer_en.pkl")
    if os.path.exists(tok_path):
        print("  英文 tokenizer 已存在，跳过训练")
    else:
        import rustbpe, tiktoken
        print("  训练英文 BPE tokenizer...")
        t0 = time.time()
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(iter(texts), vocab_size_no_special,
                                      pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks = {bytes(k): v
                           for k, v in tokenizer.get_mergeable_ranks()}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i
                          for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe_en",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        with open(tok_path, "wb") as f:
            pickle.dump(enc, f)
        print(f"  Tokenizer 训练完成，耗时 {time.time()-t0:.1f}s")

    # ── 加载 tokenizer ─────────────────────────────────
    with open(tok_path, "rb") as f:
        enc = pickle.load(f)
    bos_id = enc.encode_single_token(BOS_TOKEN)

    # ── 生成 K-fold bin 文件 ────────────────────────────
    fold_flag = os.path.join(EXP_DIR, f"folds_done_{K_FOLDS}.flag")
    if os.path.exists(fold_flag):
        print(f"  K-fold bin 文件已存在，跳过生成")
        return enc, bos_id, labels

    print(f"  生成 {K_FOLDS}-fold bin 文件...")
    labels_arr = np.array(labels)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels_arr)):
        for split, idx in [("train", train_idx), ("val", val_idx)]:
            ids_flat = []
            for i in idx:
                ids = enc.encode_ordinary(texts[i])
                ids.insert(0, bos_id)
                ids_flat.extend(ids)
            arr = np.array(ids_flat, dtype=np.int32)
            arr.tofile(os.path.join(EXP_DIR, f"{split}_fold{fold_idx}.bin"))
        print(f"    Fold {fold_idx}: "
              f"train={len(train_idx)}, val={len(val_idx)}")

    open(fold_flag, 'w').close()
    print("  K-fold bin 文件生成完毕")
    return enc, bos_id, labels


# ── 复用 prepare.py 的 make_dataloader 逻辑（指向英文目录）──

def make_en_dataloader(enc, bos_id, B, T, split, fold_idx):
    """简化版 dataloader，直接读取英文 bin 文件"""
    data_path = os.path.join(EXP_DIR, f"{split}_fold{fold_idx}.bin")
    tokens = np.fromfile(data_path, dtype=np.int32).tolist()

    row_cap = T + 1
    device_ = device

    def _batches():
        docs, current = [], []
        for tid in tokens:
            if tid == bos_id and current:
                docs.append(current)
                current = [tid]
            else:
                current.append(tid)
        if current:
            docs.append(current)
        while True:
            for i in range(0, len(docs), B):
                batch_docs = docs[i:i+B]
                rows = np.zeros((B, row_cap), dtype=np.int64)
                for r, doc in enumerate(batch_docs):
                    length = min(len(doc), row_cap)
                    rows[r, :length] = doc[:length]
                yield (torch.tensor(rows[:, :-1], device=device_),
                       torch.tensor(rows[:, 1:],  device=device_))

    return _batches()


@torch.no_grad()
def evaluate_en(model, enc, bos_id, fold_idx):
    model.eval()
    token_id_1 = enc.encode_single_token('1')
    data_path  = os.path.join(EXP_DIR, f"val_fold{fold_idx}.bin")
    tokens     = np.fromfile(data_path, dtype=np.int32).tolist()

    y_true, y_pred, y_prob = [], [], []
    current = []
    for tid in tokens:
        if tid == bos_id and current:
            if len(current) >= 2:
                context = current[:-1][-MAX_SEQ_LEN:]
                inp     = torch.tensor([context], dtype=torch.long, device=device)
                with autocast_ctx:
                    logits = model(inp)
                prob_1 = F.softmax(logits[0, -1], dim=-1)[token_id_1].item()
                pred   = 1 if logits[0, -1].argmax().item() == token_id_1 else 0
                y_true.append(1 if current[-1] == token_id_1 else 0)
                y_pred.append(pred)
                y_prob.append(prob_1)
            current = [tid]
        else:
            current.append(tid)

    if current and len(current) >= 2:
        context = current[:-1][-MAX_SEQ_LEN:]
        inp     = torch.tensor([context], dtype=torch.long, device=device)
        with autocast_ctx:
            logits  = model(inp)
        prob_1  = F.softmax(logits[0, -1], dim=-1)[token_id_1].item()
        pred    = 1 if logits[0, -1].argmax().item() == token_id_1 else 0
        y_true.append(1 if current[-1] == token_id_1 else 0)
        y_pred.append(pred)
        y_prob.append(prob_1)

    acc  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    auc  = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return auc, acc, sens, spec


def train_fold_en(fold_idx, enc, bos_id):
    print(f"\n{'='*50}")
    print(f"  英文编码 Fold {fold_idx+1}/{K_FOLDS} 训练")
    print(f"{'='*50}")

    config = build_model_config(DEPTH)
    model  = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR, embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR, adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR, weight_decay=WEIGHT_DECAY,
    )

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    grad_accum = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    loader = make_en_dataloader(enc, bos_id, DEVICE_BATCH_SIZE,
                                MAX_SEQ_LEN, "train", fold_idx)

    def sync():
        if device_type == "mps": torch.mps.synchronize()

    smooth_loss = 0
    for step in range(MAX_STEPS + 1):
        sync(); t0 = time.time()
        for _ in range(grad_accum):
            x, y = next(loader)
            with autocast_ctx:
                loss = model(x, y)
            (loss / grad_accum).backward()
            x, y = next(loader)

        progress = step / MAX_STEPS
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = get_muon_momentum(step)
                group["weight_decay"] = get_weight_decay(progress)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        smooth_loss = 0.9 * smooth_loss + 0.1 * loss.item()
        db = smooth_loss / (1 - 0.9**(step+1))
        sync(); dt = time.time() - t0
        print(f"\rFold {fold_idx+1} step {step:04d}/{MAX_STEPS} "
              f"loss={db:.4f} tok/s={int(TOTAL_BATCH_SIZE/dt):,}    ",
              end="", flush=True)

        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()

    print()
    auc, acc, sens, spec = evaluate_en(model, enc, bos_id, fold_idx)
    print(f"  Fold {fold_idx+1}: AUC={auc:.4f}  Acc={acc:.4f}  "
          f"Sens={sens:.4f}  Spec={spec:.4f}")

    del model, optimizer
    gc.collect()
    torch.mps.empty_cache()

    return {'fold': fold_idx+1, 'auc': auc, 'acc': acc,
            'sensitivity': sens, 'specificity': spec}


def main():
    print("\n实验 B：英文 vs 中文叙述编码 对比实验")
    print("原论文中文结果：AUC = 0.762 ± 0.070\n")

    enc, bos_id, _ = prepare_english_data()

    print("\n── 步骤 2：英文编码 5-Fold CV 训练 ───────────────")
    results = []
    for fold_idx in range(K_FOLDS):
        r = train_fold_en(fold_idx, enc, bos_id)
        results.append(r)

    # ── 汇总 ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  英文编码 5-Fold CV 结果")
    print(f"{'='*50}")

    for r in results:
        print(f"  Fold {r['fold']}: AUC={r['auc']:.4f}  Acc={r['acc']:.4f}  "
              f"Sens={r['sensitivity']:.4f}  Spec={r['specificity']:.4f}")

    mean_auc  = np.mean([r['auc']  for r in results])
    std_auc   = np.std( [r['auc']  for r in results], ddof=1)
    mean_acc  = np.mean([r['acc']  for r in results])
    std_acc   = np.std( [r['acc']  for r in results], ddof=1)
    mean_sens = np.mean([r['sensitivity'] for r in results])
    std_sens  = np.std( [r['sensitivity'] for r in results], ddof=1)
    mean_spec = np.mean([r['specificity'] for r in results])
    std_spec  = np.std( [r['specificity'] for r in results], ddof=1)

    print(f"\n  英文编码 Mean±SD:")
    print(f"    AUC         = {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"    Accuracy    = {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"    Sensitivity = {mean_sens:.3f} ± {std_sens:.3f}")
    print(f"    Specificity = {mean_spec:.3f} ± {std_spec:.3f}")

    print(f"\n  对比：")
    print(f"    中文编码 AUC = 0.762 ± 0.070（原论文）")
    print(f"    英文编码 AUC = {mean_auc:.3f} ± {std_auc:.3f}（本实验）")
    print(f"    差值 ΔAUC    = {mean_auc - 0.762:+.3f}")

    # ── 保存结果 ─────────────────────────────────────────
    output = {
        'experiment': 'B_language_comparison',
        'english_encoding': {
            'fold_results': results,
            'summary': {
                'auc':  {'mean': mean_auc,  'std': std_auc},
                'acc':  {'mean': mean_acc,  'std': std_acc},
                'sensitivity': {'mean': mean_sens, 'std': std_sens},
                'specificity': {'mean': mean_spec, 'std': std_spec},
            }
        },
        'chinese_encoding_reference': {
            'auc': {'mean': 0.762, 'std': 0.070},
            'source': 'Original paper, same architecture and K-fold protocol'
        }
    }
    with open('experiment_B_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✅ 结果已保存至 experiment_B_results.json")


if __name__ == "__main__":
    main()
