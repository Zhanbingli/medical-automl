"""
实验 A：零填充 vs 均值填充 对比实验
===================================
验证核心假设：零填充是导致 Specificity Collapse 的主因

使用已训练好的 5 个 fold 模型，分别用两种填充策略做外部验证：
  - 策略1：零填充（Zero-padding，原始方案）
  - 策略2：均值填充（Mean imputation，ca=0.67, thal=4.73，UCI训练集均值）

运行方式：
    python experiment_A_imputation.py

依赖：saved_models/ 目录下已有 model_fold0.pt ~ model_fold4.pt
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

# ── 环境检查 ────────────────────────────────────────────────
if sys.platform != "darwin":
    raise RuntimeError("This script requires macOS with MPS.")
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS not available.")

from prepare import Tokenizer, MAX_SEQ_LEN, DATA_DIR, TOKENIZER_DIR, BOS_TOKEN
from train import GPTConfig, GPT, build_model_config, device, autocast_ctx
from train import DEPTH

# ── Kaggle 数据路径（与 plot_dataset_shift.py 一致）──────────
KAGGLE_PATH = os.path.expanduser(
    "~/.cache/kagglehub/datasets/fedesoriano/"
    "heart-failure-prediction/versions/1/heart.csv"
)

# ── UCI 训练集中 ca 和 thal 的均值（用于均值填充）────────────
# 计算方式：patients.csv 中去除 '?' 后的有效值均值
UCI_CA_MEAN   = 0.67   # ca 字段均值（0~3）
UCI_THAL_MEAN = 4.73   # thal 字段均值（3=normal,6=fixed,7=reversable）

# ────────────────────────────────────────────────────────────

def load_kaggle_data(ca_fill_value, thal_fill_value):
    """
    加载并对齐 Kaggle 数据，ca 和 thal 使用指定值填充。
    返回：texts（列表），labels（列表）
    """
    import pandas as pd

    df = pd.read_csv(KAGGLE_PATH)

    # 列名对齐
    col_map = {
        'Age': 'age', 'RestingBP': 'trestbps', 'Cholesterol': 'chol',
        'FastingBS': 'fbs', 'MaxHR': 'thalach', 'Oldpeak': 'oldpeak',
        'HeartDisease': 'num_binary'
    }
    df = df.rename(columns=col_map)

    # 分类编码
    df['sex']    = df['Sex'].map({'M': 1, 'F': 0})
    df['cp']     = df['ChestPainType'].map({'TA': 1, 'ATA': 2, 'NAP': 3, 'ASY': 4})
    df['restecg'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    df['exang']  = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df['slope']  = df['ST_Slope'].map({'Up': 1, 'Flat': 2, 'Down': 3})

    # 填充缺失特征
    df['ca']   = ca_fill_value
    df['thal'] = thal_fill_value

    texts  = []
    labels = []
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
        labels.append(int(row['num_binary']))

    return texts, labels


@torch.no_grad()
def evaluate_on_texts(model, tokenizer, texts, labels):
    """对文本列表做推断，返回 AUC, Accuracy, Sensitivity, Specificity"""
    model.eval()
    token_id_1 = tokenizer.enc.encode_single_token('1')
    bos_id     = tokenizer.get_bos_token_id()

    y_true, y_pred, y_prob = [], [], []

    for text, label in zip(texts, labels):
        ids = tokenizer.encode(text, prepend=bos_id)
        # 去掉最后一个 token（label token），用前面作为 context
        context = ids[:-1]
        if len(context) > MAX_SEQ_LEN:
            context = context[-MAX_SEQ_LEN:]

        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        with autocast_ctx:
            logits = model(input_tensor)

        probs  = F.softmax(logits[0, -1], dim=-1)
        prob_1 = probs[token_id_1].item()
        pred   = 1 if logits[0, -1].argmax().item() == token_id_1 else 0

        y_true.append(label)
        y_pred.append(pred)
        y_prob.append(prob_1)

    acc  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    auc  = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return auc, acc, sens, spec


def run_strategy(strategy_name, ca_fill, thal_fill, tokenizer):
    print(f"\n{'='*55}")
    print(f"  策略：{strategy_name}  (ca={ca_fill}, thal={thal_fill})")
    print(f"{'='*55}")

    texts, labels = load_kaggle_data(ca_fill, thal_fill)
    print(f"  加载 Kaggle 数据：{len(texts)} 条")

    fold_results = []
    for fold_idx in range(5):
        model_path = f"saved_models/model_fold{fold_idx}.pt"
        assert os.path.exists(model_path), f"找不到模型：{model_path}"

        config = build_model_config(DEPTH)
        model  = GPT(config).to(device)
        state  = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        auc, acc, sens, spec = evaluate_on_texts(model, tokenizer, texts, labels)
        fold_results.append({'fold': fold_idx + 1,
                             'auc': auc, 'acc': acc,
                             'sensitivity': sens, 'specificity': spec})
        print(f"  Fold {fold_idx+1}: AUC={auc:.3f}  Acc={acc:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}")

        del model
        torch.mps.empty_cache()

    mean_auc  = np.mean([r['auc']  for r in fold_results])
    std_auc   = np.std( [r['auc']  for r in fold_results], ddof=1)
    mean_acc  = np.mean([r['acc']  for r in fold_results])
    std_acc   = np.std( [r['acc']  for r in fold_results], ddof=1)
    mean_sens = np.mean([r['sensitivity'] for r in fold_results])
    std_sens  = np.std( [r['sensitivity'] for r in fold_results], ddof=1)
    mean_spec = np.mean([r['specificity'] for r in fold_results])
    std_spec  = np.std( [r['specificity'] for r in fold_results], ddof=1)

    print(f"\n  Mean±SD:")
    print(f"    AUC         = {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"    Accuracy    = {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"    Sensitivity = {mean_sens:.3f} ± {std_sens:.3f}")
    print(f"    Specificity = {mean_spec:.3f} ± {std_spec:.3f}")

    return {
        'strategy': strategy_name,
        'ca_fill': ca_fill, 'thal_fill': thal_fill,
        'fold_results': fold_results,
        'summary': {
            'auc':  {'mean': mean_auc,  'std': std_auc},
            'acc':  {'mean': mean_acc,  'std': std_acc},
            'sensitivity': {'mean': mean_sens, 'std': std_sens},
            'specificity': {'mean': mean_spec, 'std': std_spec},
        }
    }


def main():
    print("\n实验 A：零填充 vs 均值填充 对比实验")
    print("目的：验证零填充是导致 Specificity Collapse 的主因\n")

    tokenizer = Tokenizer.from_directory()

    # 策略1：零填充（原论文方案）
    result_zero = run_strategy(
        strategy_name="零填充 (Zero-padding，原始方案)",
        ca_fill=0, thal_fill=0,
        tokenizer=tokenizer
    )

    # 策略2：均值填充
    result_mean = run_strategy(
        strategy_name="均值填充 (Mean imputation)",
        ca_fill=UCI_CA_MEAN, thal_fill=UCI_THAL_MEAN,
        tokenizer=tokenizer
    )

    # ── 结果对比 ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  最终对比结果")
    print(f"{'='*55}")
    print(f"  {'指标':<14} {'零填充':>14} {'均值填充':>14} {'差值 (均值-零)':>16}")
    print(f"  {'-'*58}")
    for metric in ['auc', 'acc', 'sensitivity', 'specificity']:
        z = result_zero['summary'][metric]
        m = result_mean['summary'][metric]
        diff = m['mean'] - z['mean']
        print(f"  {metric:<14} "
              f"{z['mean']:.3f}±{z['std']:.3f}   "
              f"{m['mean']:.3f}±{m['std']:.3f}   "
              f"{diff:+.3f}")

    # ── 保存结果 ──────────────────────────────────────────
    output = {
        'experiment': 'A_imputation_comparison',
        'zero_padding': result_zero,
        'mean_imputation': result_mean
    }
    with open('experiment_A_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n✅ 结果已保存至 experiment_A_results.json")


if __name__ == "__main__":
    main()
