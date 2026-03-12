"""
Medical AutoML - K-Fold Data Preparation

Prepares data for K-fold cross validation by creating K train/validation splits.
Each fold uses a different subset of data for validation.

Usage: python prepare_kfold.py --k_folds 5
"""

import os
import sys
import argparse
import pickle
import random
from sklearn.model_selection import StratifiedKFold
import numpy as np
import numpy as np
import torch

# Import from prepare.py
from prepare import (
    DATA_DIR, TOKENIZER_DIR, CSV_PATH, BOS_TOKEN, MAX_SEQ_LEN,
    load_and_process_patients, train_tokenizer, Tokenizer
)

def split_data_kfold(texts, k_folds=5, seed=42):
    """Split data into K folds for cross validation."""
    labels = [int(text.split('最终诊断结果为：')[-1].strip()[0]) for text in texts]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    folds = []

    texts_np = np.array(texts)
    labels_np = np.array(labels)

    for i, (train_idx, val_idx) in enumerate(skf.split(texts_np, labels_np)):
        train_texts = texts_np[train_idx].tolist()
        val_texts = texts_np[val_idx].tolist()

        folds.append({
            'fold': i,
            'train': train_texts,
            'val': val_texts,
            'train_size': len(train_texts),
            'val_size': len(val_texts)
        })
    return folds
def encode_and_save_fold(fold_data, tokenizer, fold_idx):
    """Encode and save train/val data for a specific fold."""
    train_path = os.path.join(DATA_DIR, f"train_fold{fold_idx}.bin")
    val_path = os.path.join(DATA_DIR, f"val_fold{fold_idx}.bin")

    bos_token_id = tokenizer.get_bos_token_id()

    # Encode train data
    print(f"  Encoding fold {fold_idx} train data ({fold_data['train_size']} samples)...")
    train_ids = []
    for text in fold_data['train']:
        ids = tokenizer.enc.encode_ordinary(text)
        ids.insert(0, bos_token_id)
        train_ids.extend(ids)

    # Encode val data
    print(f"  Encoding fold {fold_idx} val data ({fold_data['val_size']} samples)...")
    val_ids = []
    for text in fold_data['val']:
        ids = tokenizer.enc.encode_ordinary(text)
        ids.insert(0, bos_token_id)
        val_ids.extend(ids)

    # Save as binary
    train_array = torch.tensor(train_ids, dtype=torch.int32)
    val_array = torch.tensor(val_ids, dtype=torch.int32)

    train_array.numpy().tofile(train_path)
    val_array.numpy().tofile(val_path)

    print(f"  Saved fold {fold_idx}: train ({len(train_ids)} tokens), val ({len(val_ids)} tokens)")

    return train_path, val_path

def prepare_kfold_data(k_folds=5, seed=42):
    """Prepare all K-fold data splits."""
    print(f"\n{'='*60}")
    print(f"Preparing {k_folds}-Fold Cross Validation Data")
    print(f"{'='*60}\n")

    # Load and process data
    print("Loading patients.csv...")
    texts = load_and_process_patients()
    print(f"Total samples: {len(texts)}\n")

    # Create K folds
    print(f"Creating {k_folds} folds with seed {seed}...")
    folds = split_data_kfold(texts, k_folds, seed)

    print(f"\nFold Distribution:")
    print(f"{'Fold':<6} {'Train Size':<12} {'Val Size':<12} {'Val %':<8}")
    print("-" * 40)
    for fold in folds:
        val_pct = 100 * fold['val_size'] / len(texts)
        print(f"{fold['fold']:<6} {fold['train_size']:<12} {fold['val_size']:<12} {val_pct:<8.1f}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    with open(os.path.join(TOKENIZER_DIR, "tokenizer.pkl"), "rb") as f:
        enc = pickle.load(f)
    tokenizer = Tokenizer(enc)

    # Encode and save each fold
    print(f"\nEncoding and saving fold data...")
    for fold_data in folds:
        encode_and_save_fold(fold_data, tokenizer, fold_data['fold'])

    # Save fold information
    fold_info_path = os.path.join(DATA_DIR, f"kfold_{k_folds}_info.txt")
    with open(fold_info_path, 'w') as f:
        f.write(f"K-Fold Cross Validation Information\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"K={k_folds}, Seed={seed}\n")
        f.write(f"Total samples: {len(texts)}\n\n")
        f.write(f"Fold Distribution:\n")
        for fold in folds:
            f.write(f"  Fold {fold['fold']}: Train={fold['train_size']}, Val={fold['val_size']}\n")

    print(f"\nFold information saved to: {fold_info_path}")
    print(f"\n{'='*60}")
    print(f"K-Fold Data Preparation Complete!")
    print(f"{'='*60}\n")

    return folds

def main():
    parser = argparse.ArgumentParser(description='Prepare K-fold cross validation data')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Prepare regular data first (if not exists)
    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found. Training tokenizer first...")
        train_tokenizer(load_and_process_patients())

    # Prepare K-fold data
    prepare_kfold_data(k_folds=args.k_folds, seed=args.seed)

if __name__ == "__main__":
    main()
