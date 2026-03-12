"""
Medical AutoML - Data Preparation Module

Handles data preprocessing, BPE tokenizer training, and clinical metrics evaluation
for cardiovascular disease diagnosis using transformer models.

Part of the medical-automl project: https://github.com/Zhanbingli/medical-automl

Usage:
    python prepare.py

This module performs:
- Structured patient data textualization
- Custom BPE tokenizer training (8K vocab for medical Chinese)
- Binary data encoding for efficient training
- Clinical metrics calculation (AUC, Sensitivity, Specificity)
"""

import os
import sys
import time
import pickle
from multiprocessing import Pool

import pandas as pd
import numpy as np
import rustbpe
import tiktoken
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np

def verify_macos_env():
    import sys
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()

verify_macos_env()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patients.csv")
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def load_and_process_patients():
    """Load patients.csv and convert to structured text."""
    print("Loading patients.csv...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} patient records")

    df['num_binary'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

    texts = []
    for _, row in df.iterrows():
        text = (
            f"患者特征：年龄{row['age']}，性别{row['sex']}，"
            f"胸痛类型{row['cp']}，静息血压{row['trestbps']}，"
            f"胆固醇{row['chol']}，空腹血糖{row['fbs']}，"
            f"静息心电图{row['restecg']}，最大心率{row['thalach']}，"
            f"运动诱发心绞痛{row['exang']}，ST段压低{row['oldpeak']}，"
            f"ST段斜率{row['slope']}，透视血管数{row['ca']}，"
            f"地中海贫血{row['thal']}。最终诊断结果为：{row['num_binary']}"
        )
        texts.append(text)

    print(f"Generated {len(texts)} text samples")
    return texts


def split_data(texts, val_ratio=0.1):
    """Split data into train and validation sets."""
    val_size = int(len(texts) * val_ratio)
    train_size = len(texts) - val_size

    train_texts = texts[:train_size]
    val_texts = texts[train_size:]

    print(f"Split: {len(train_texts)} train, {len(val_texts)} validation")
    return train_texts, val_texts


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(all_texts):
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

    if os.path.exists(tokenizer_pkl):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter(all_texts), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    test = "患者特征：年龄63，性别1，胸痛类型1"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Encode and save data
# ---------------------------------------------------------------------------

def encode_and_save():
    """Encode texts and save train.bin/val.bin."""
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Data: train.bin and val.bin already exist at {DATA_DIR}")
        return

    print("Loading tokenizer...")
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "rb") as f:
        enc = pickle.load(f)

    texts = load_and_process_patients()
    train_texts, val_texts = split_data(texts, val_ratio=0.1)

    print("Encoding train data...")
    bos_token_id = enc.encode_single_token(BOS_TOKEN)
    train_ids = []
    for text in train_texts:
        ids = enc.encode_ordinary(text)
        ids.insert(0, bos_token_id)
        train_ids.extend(ids)

    print("Encoding val data...")
    val_ids = []
    for text in val_texts:
        ids = enc.encode_ordinary(text)
        ids.insert(0, bos_token_id)
        val_ids.extend(ids)

    train_array = torch.tensor(train_ids, dtype=torch.int32)
    val_array = torch.tensor(val_ids, dtype=torch.int32)

    train_array.numpy().tofile(train_path)
    val_array.numpy().tofile(val_path)

    print(f"Saved train.bin ({len(train_ids)} tokens) to {train_path}")
    print(f"Saved val.bin ({len(val_ids)} tokens) to {val_path}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from binary files.
    
    Supports both regular splits ("train", "val") and K-fold splits ("train_fold0", "val_fold0", etc.)
    """
    # Check if this is a K-fold split
    if "_fold" in split:
        # Parse fold number from split name (e.g., "train_fold0" -> "train_fold0.bin")
        data_path = os.path.join(DATA_DIR, f"{split}.bin")
    else:
        # Regular split
        train_path = os.path.join(DATA_DIR, "train.bin")
        val_path = os.path.join(DATA_DIR, "val.bin")
        data_path = train_path if split == "train" else val_path
    
    assert os.path.exists(data_path), f"Data file not found: {data_path}. Run prepare.py or prepare_kfold.py first."

    tokens = np.fromfile(data_path, dtype=np.int32).tolist()
    epoch = 1
    while True:
        token_lists = []
        current_list = []
        for token_id in tokens:
            current_list.append(token_id)
            if token_id == bos_token_id and len(current_list) > 1:
                token_lists.append(current_list)
                current_list = []
        if current_list:
            token_lists.append(current_list)

        for i in range(0, len(token_lists), tokenizer_batch_size):
            yield token_lists[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    
    Supports both regular splits ("train", "val") and K-fold splits ("train_fold0", "val_fold0", etc.)
    """
    # Support regular splits and K-fold splits
    valid_splits = ["train", "val"]
    for i in range(10):
        valid_splits.append("train_fold" + str(i))
        valid_splits.append("val_fold" + str(i))
    if split not in valid_splits:
        raise ValueError("Invalid split: " + split + ". Must be 'train', 'val', or K-fold split like 'train_fold0'")
    row_capacity = T + 1
    global bos_token_id
    bos_token_id = tokenizer.get_bos_token_id()
    batches = _document_batches(split)
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        doc_buffer.extend(doc_batch)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=(device=="cuda"))
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clinical_metrics(model, tokenizer, batch_size):
    """
    Calculate Validation Accuracy, AUC, Sensitivity, and Specificity
    by analyzing the model's output probabilities.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, confusion_matrix
    import torch.nn.functional as F

    device = next(model.parameters()).device
    val_path = os.path.join(DATA_DIR, "val.bin")
    val_tokens = np.fromfile(val_path, dtype=np.int32).tolist()

    # 获取数字 '1' 的 Token ID，用于提取患病概率
    try:
        token_id_1 = tokenizer.enc.encode_single_token('1')
    except:
        token_id_1 = tokenizer.enc.encode_ordinary('1')[-1]

    y_true = []
    y_pred = []
    y_prob = []

    current_doc = []
    for token_id in val_tokens:
        if token_id == bos_token_id and current_doc:
            if len(current_doc) >= 2:
                last_token = current_doc[-1]
                context = current_doc[:-1]

                if len(context) > MAX_SEQ_LEN:
                    context = context[-MAX_SEQ_LEN:]

                input_tensor = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(input_tensor)

                # 计算属于 '1' (患病) 的概率
                probs = F.softmax(logits[0, -1], dim=-1)
                prob_1 = probs[token_id_1].item()

                # 获取硬预测分类
                pred_token = logits[0, -1].argmax().item()

                # 临床转化
                actual_label = 1 if last_token == token_id_1 else 0
                pred_label = 1 if pred_token == token_id_1 else 0

                y_true.append(actual_label)
                y_pred.append(pred_label)
                y_prob.append(prob_1)

            current_doc = [token_id]
        else:
            current_doc.append(token_id)

    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 计算铁三角指标
    accuracy = sum([1 for t, p in zip(y_true, y_pred) if t == p]) / len(y_true)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5 # 防止 batch 里只有一个类别的极端情况

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"诊断混淆矩阵: 真阳性(TP)={tp}, 假阴性(漏诊/FN)={fn}, 真阴性(TN)={tn}, 假阳性(误诊/FP)={fp}")
    return accuracy, auc, sensitivity, specificity
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    texts = load_and_process_patients()
    train_texts, val_texts = split_data(texts, val_ratio=0.1)
    all_texts = train_texts + val_texts

    train_tokenizer(all_texts)
    print()

    encode_and_save()
    print()
    print("Done! Ready to train.")
