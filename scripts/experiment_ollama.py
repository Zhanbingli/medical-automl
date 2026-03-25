"""
scripts/experiment_ollama.py

Ollama + Qwen3.5:2B  Zero-shot 和 5-shot 对比实验
  - UCI Heart Disease: 5-fold CV（与其他所有实验相同的 StratifiedKFold seed=42）
  - Kaggle Heart Failure: 外部验证（ca/thal 零填充，与 Transformer 管线完全一致）

研究问题：
  预训练 LLM 是否同样存在 Transformer 的泛化性问题（内部验证 AUC >> 外部验证 AUC）？
  LLM 在 Kaggle 上是否也出现 Specificity Collapse？

Requirements:
  - Ollama 已在本地运行: `ollama serve`
  - 模型已拉取:          `ollama pull qwen3.5:2b`
  - patients.csv 在项目根目录
  - Kaggle 数据集路径正确（KAGGLE_CSV 见下方）

Usage:
  uv run python scripts/experiment_ollama.py

  快速测试（每个 split 只跑前 N 条）:
  QUICK=30 uv run python scripts/experiment_ollama.py
"""

import os
import sys
import json
import time
import random
import re
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# ── 配置 ──────────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "qwen3.5:2b"   # 如已拉取不同名字请修改
TEMPERATURE     = 0.0
MAX_TOKENS      = 512   # Ollama think:false 时 JSON ~20 token 够用；留余量防 thinking 泄漏
WORKERS         = 1     # Ollama 单线程处理，并发只会造成排队超时
K_FOLDS         = 5
RANDOM_STATE    = 42
N_SHOT          = 5
REQUEST_TIMEOUT = 120   # 单次请求超时（秒）

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(ROOT_DIR, "patients.csv")
KAGGLE_CSV  = os.path.expanduser(
    "~/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv"
)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "experiment_ollama_results.json")

# 快速模式：QUICK=N 时每个 split 最多跑 N 条（用于调试）
QUICK = int(os.environ.get("QUICK", "0"))

# ── 特征解码 ──────────────────────────────────────────────────────────────────
SEX_DEC   = {0: "Female", 1: "Male"}
CP_DEC    = {1: "typical angina", 2: "atypical angina",
             3: "non-anginal pain", 4: "asymptomatic"}
FBS_DEC   = {0: "<=120 mg/dl (normal)", 1: ">120 mg/dl (elevated)"}
ECG_DEC   = {0: "normal", 1: "ST-T wave abnormality",
             2: "left ventricular hypertrophy"}
EXANG_DEC = {0: "No", 1: "Yes"}
SLP_DEC   = {0: "unknown", 1: "upsloping", 2: "flat", 3: "downsloping"}
THAL_DEC  = {0: "not available", 3: "normal",
             6: "fixed defect", 7: "reversible defect"}


def _get(d, key, decode_map=None):
    v = d.get(key, 0)
    if decode_map is not None:
        return decode_map.get(int(v), str(v))
    return v


def format_patient(row: dict) -> str:
    """把患者 dict 转为 LLM 可读的英文描述（不含 label）。"""
    return (
        f"Age: {row['age']}, "
        f"Sex: {_get(row,'sex',SEX_DEC)}, "
        f"Chest pain type: {_get(row,'cp',CP_DEC)}, "
        f"Resting BP: {row['trestbps']} mmHg, "
        f"Cholesterol: {row['chol']} mg/dl, "
        f"Fasting blood sugar: {_get(row,'fbs',FBS_DEC)}, "
        f"Resting ECG: {_get(row,'restecg',ECG_DEC)}, "
        f"Max heart rate: {row['thalach']} bpm, "
        f"Exercise-induced angina: {_get(row,'exang',EXANG_DEC)}, "
        f"ST depression (oldpeak): {row['oldpeak']}, "
        f"ST slope: {_get(row,'slope',SLP_DEC)}, "
        f"Fluoroscopy vessels colored (ca): {row['ca']}, "
        f"Thalassemia (thal): {_get(row,'thal',THAL_DEC)}"
    )


# ── Prompt 构建 ───────────────────────────────────────────────────────────────
_TASK_DESC = (
    "You are a clinical decision support AI. "
    "Assess the probability that the patient has heart disease "
    "based on the given features.\n"
    "Output ONLY a JSON object, no other text:\n"
    '{"probability": <float between 0.0 and 1.0>}'
)


def build_zero_shot_prompt(patient_text: str) -> str:
    return f"{_TASK_DESC}\n\nPatient: {patient_text}\n\nOutput:"


def build_few_shot_prompt(examples: list[dict], patient_text: str) -> str:
    ex_block = "\n\n".join(
        f"Example {i+1}: {ex['text']}\n"
        f"Diagnosis: {'heart disease present (1)' if ex['label'] == 1 else 'no heart disease (0)'}"
        for i, ex in enumerate(examples)
    )
    return (
        f"{_TASK_DESC}\n\n"
        f"Here are {len(examples)} labeled examples:\n\n"
        f"{ex_block}\n\n"
        f"Now assess this patient: {patient_text}\n\nOutput:"
    )


# ── Ollama 调用与概率解析 ─────────────────────────────────────────────────────
def call_ollama(prompt: str, retries: int = 3) -> float | None:
    """调用 Ollama，返回 0~1 的概率浮点数，失败返回 None。"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,     # Ollama ≥0.7: 关闭 Qwen3 thinking 模式
        "keep_alive": -1,   # 保持模型常驻内存
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()

            # 优先解析 JSON {"probability": x}
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    obj  = json.loads(text[start:end])
                    prob = float(obj.get("probability", -1))
                    if 0.0 <= prob <= 1.0:
                        return prob
                except (json.JSONDecodeError, ValueError):
                    pass

            # 后备：找第一个 0~1 之间的小数
            nums = re.findall(r'\b(?:0\.\d+|1\.0+|0|1)\b', text)
            for n in nums:
                prob = float(n)
                if 0.0 <= prob <= 1.0:
                    return prob

            # 解析失败：打印原始回复帮助诊断（只打前 200 字符）
            print(f"\n      [debug] parse failed, raw response: {repr(text[:200])}")

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"      [!] attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2)
        except Exception as e:
            print(f"      [!] error (attempt {attempt+1}): {e}")

    return None


# ── Few-shot 示例选取 ─────────────────────────────────────────────────────────
def pick_examples(train_rows: list[dict], n: int = N_SHOT,
                  seed: int = RANDOM_STATE) -> list[dict]:
    """从训练集中均衡采样 n 个示例（含预格式化的 text 字段）。"""
    rng = random.Random(seed)
    pos = [r for r in train_rows if r["label"] == 1]
    neg = [r for r in train_rows if r["label"] == 0]
    n_pos = n // 2
    n_neg = n - n_pos
    chosen = (rng.sample(pos, min(n_pos, len(pos))) +
              rng.sample(neg, min(n_neg, len(neg))))
    rng.shuffle(chosen)
    for ex in chosen:
        ex["text"] = format_patient(ex)
    return chosen


# ── 批量推理 ──────────────────────────────────────────────────────────────────
def _infer_one(args: tuple) -> tuple[int, int, float | None]:
    """单条推理，供 ThreadPoolExecutor 调用。"""
    idx, rec, prompt = args
    return idx, rec["label"], call_ollama(prompt)


def run_inference(records: list[dict],
                  examples: list[dict] | None = None,
                  tag: str = "") -> dict:
    """对一批记录并行运行 LLM 推理，返回指标 dict。"""
    if QUICK:
        records = records[:QUICK]

    # 预先构建所有 prompt（无 IO，极快）
    tasks = []
    for i, rec in enumerate(records):
        patient_text = format_patient(rec)
        prompt = (build_few_shot_prompt(examples, patient_text)
                  if examples else
                  build_zero_shot_prompt(patient_text))
        tasks.append((i, rec, prompt))

    slot = [None] * len(records)  # 按原始顺序存结果
    n_failed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fut_map = {pool.submit(_infer_one, t): t[0] for t in tasks}
        with tqdm(total=len(records), desc=f"  {tag}", unit="rec",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}, {rate_fmt}]{postfix}") as pbar:
            for fut in as_completed(fut_map):
                idx, label, prob = fut.result()
                if prob is None:
                    prob = 0.5
                    n_failed += 1
                slot[idx] = (label, prob)
                pbar.update(1)
                pbar.set_postfix(failed=n_failed, last_p=f"{prob:.2f}")

    y_true = [s[0] for s in slot]
    y_prob = [s[1] for s in slot]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    auc = (roc_auc_score(y_true, y_prob)
           if len(set(y_true)) >= 2 else float("nan"))
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc), "acc": float(acc),
        "sensitivity": float(sens), "specificity": float(spec),
        "n_records": len(records), "n_failed": n_failed,
        "elapsed_s": round(time.time() - t0, 1),
    }


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def _safe_int(v, default=0):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def load_uci() -> list[dict]:
    df = pd.read_csv(CSV_PATH)
    df["num_binary"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "age": int(row["age"]), "sex": int(row["sex"]),
            "cp": int(row["cp"]), "trestbps": int(row["trestbps"]),
            "chol": int(row["chol"]), "fbs": int(row["fbs"]),
            "restecg": int(row["restecg"]), "thalach": int(row["thalach"]),
            "exang": int(row["exang"]), "oldpeak": float(row["oldpeak"]),
            "slope": int(row["slope"]),
            "ca":   _safe_int(row["ca"],   default=0),
            "thal": _safe_int(row["thal"], default=0),
            "label": int(row["num_binary"]),
        })
    return rows


def load_kaggle() -> list[dict]:
    SEX = {"M": 1, "F": 0}
    CP  = {"TA": 1, "ATA": 2, "NAP": 3, "ASY": 4}
    ECG = {"Normal": 0, "ST": 1, "LVH": 2}
    ANG = {"Y": 1, "N": 0}
    SLP = {"Up": 1, "Flat": 2, "Down": 3}

    df = pd.read_csv(KAGGLE_CSV)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "age": int(row["Age"]),
            "sex": SEX.get(str(row["Sex"]), 0),
            "cp":  CP.get(str(row["ChestPainType"]), 0),
            "trestbps": int(row["RestingBP"]),
            "chol": int(row["Cholesterol"]),
            "fbs": int(row["FastingBS"]),
            "restecg": ECG.get(str(row["RestingECG"]), 0),
            "thalach": int(row["MaxHR"]),
            "exang": ANG.get(str(row["ExerciseAngina"]), 0),
            "oldpeak": float(row["Oldpeak"]),
            "slope": SLP.get(str(row["ST_Slope"]), 0),
            "ca":   0,   # Kaggle 无此特征 → 零填充（与 Transformer 管线一致）
            "thal": 0,   # 同上
            "label": int(row["HeartDisease"]),
        })
    return rows


# ── 汇总 folds ────────────────────────────────────────────────────────────────
def summarize(fold_results: list[dict]) -> dict:
    summary = {}
    for m in ["auc", "acc", "sensitivity", "specificity"]:
        vals = [r[m] for r in fold_results if not np.isnan(r[m])]
        summary[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return summary


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"Experiment: Ollama LLM  Zero-shot vs {N_SHOT}-shot")
    print(f"Model : {MODEL}")
    print(f"Splits: {K_FOLDS}-fold CV (UCI) + Kaggle external validation")
    if QUICK:
        print(f"[QUICK MODE] max {QUICK} samples per split")
    print("=" * 70)

    # ── 检查 Ollama ────────────────────────────────────────────────────────────
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        print(f"\nOllama 已连接。可用模型: {available}")
        if not any(MODEL.split(":")[0] in m for m in available):
            print(f"  [!] '{MODEL}' 未找到。请先运行: ollama pull {MODEL}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 无法连接 Ollama: {e}")
        print("  请先启动: ollama serve")
        sys.exit(1)

    # ── 加载数据 ───────────────────────────────────────────────────────────────
    print("\n[1/3] 加载数据...")
    uci_rows = load_uci()
    print(f"  UCI   : {len(uci_rows)} 条  "
          f"(正例={sum(r['label']==1 for r in uci_rows)}, "
          f"负例={sum(r['label']==0 for r in uci_rows)})")

    kaggle_rows = None
    if os.path.exists(KAGGLE_CSV):
        kaggle_rows = load_kaggle()
        print(f"  Kaggle: {len(kaggle_rows)} 条  "
              f"(正例={sum(r['label']==1 for r in kaggle_rows)}, "
              f"负例={sum(r['label']==0 for r in kaggle_rows)})")
        print("  注: Kaggle 中 ca/thal 不存在 → 零填充（与 Transformer 管线相同）")
    else:
        print(f"  [!] Kaggle CSV 未找到: {KAGGLE_CSV}")
        print("      请先运行: uv run python download_data.py")

    # ── UCI 5-fold CV ──────────────────────────────────────────────────────────
    print(f"\n[2/3] UCI 5-fold CV  (StratifiedKFold seed={RANDOM_STATE})...")

    labels_arr = np.array([r["label"] for r in uci_rows])
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    # ── 断点续跑：从已有检查点恢复 ───────────────────────────────────────────────
    CKPT_JSON = os.path.join(RESULTS_DIR, "experiment_ollama_checkpoint.json")

    def save_ckpt(zs_folds, fs_folds):
        with open(CKPT_JSON, "w") as f:
            json.dump({"zs_folds": zs_folds, "fs_folds": fs_folds}, f, indent=2)

    def load_ckpt():
        if os.path.exists(CKPT_JSON):
            with open(CKPT_JSON) as f:
                d = json.load(f)
            return d["zs_folds"], d["fs_folds"]
        return [], []

    zs_folds, fs_folds = load_ckpt()
    done_zs = {r["fold"] for r in zs_folds}   # 已完成 zero-shot 的 fold
    done_fs = {r["fold"] for r in fs_folds}    # 已完成 5-shot 的 fold

    if done_zs:
        print(f"  ✓ 从检查点恢复  ZS完成: {sorted(done_zs)}  FS完成: {sorted(done_fs)}")

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(uci_rows, labels_arr)):

        if fold_idx in done_zs and fold_idx in done_fs:
            print(f"\n  ── Fold {fold_idx}  跳过（ZS+FS 均已完成）──────────────")
            continue

        train_rows = [uci_rows[i] for i in train_idx]
        val_rows   = [uci_rows[i] for i in val_idx]
        n_val = min(len(val_rows), QUICK) if QUICK else len(val_rows)

        print(f"\n  ── Fold {fold_idx}  "
              f"(train={len(train_rows)}, val={n_val}) ──────────────")

        # Zero-shot（如果已完成则跳过）
        if fold_idx not in done_zs:
            print(f"    Zero-shot ({n_val} records)...")
            zs = run_inference(val_rows, examples=None,
                               tag=f"fold{fold_idx}_zs")
            zs_folds.append({"fold": fold_idx, **zs})
            save_ckpt(zs_folds, fs_folds)          # ZS 完成立即保存
            print(f"    ZS  AUC={zs['auc']:.4f}  "
                  f"Sens={zs['sensitivity']:.4f}  "
                  f"Spec={zs['specificity']:.4f}  "
                  f"(failed={zs['n_failed']}/{zs['n_records']})")
        else:
            print(f"    Zero-shot 跳过（已有结果）")

        # 5-shot（如果已完成则跳过）
        if fold_idx not in done_fs:
            examples = pick_examples(train_rows, n=N_SHOT,
                                      seed=RANDOM_STATE + fold_idx)
            print(f"    {N_SHOT}-shot ({n_val} records)  "
                  f"examples: {[ex['label'] for ex in examples]}...")
            fs = run_inference(val_rows, examples=examples,
                               tag=f"fold{fold_idx}_fs")
            fs_folds.append({"fold": fold_idx, **fs})
            save_ckpt(zs_folds, fs_folds)          # FS 完成立即保存
            print(f"    FS  AUC={fs['auc']:.4f}  "
                  f"Sens={fs['sensitivity']:.4f}  "
                  f"Spec={fs['specificity']:.4f}  "
                  f"(failed={fs['n_failed']}/{fs['n_records']})")
        else:
            print(f"    {N_SHOT}-shot 跳过（已有结果）")

    zs_uci = summarize(zs_folds)
    fs_uci = summarize(fs_folds)

    # ── Kaggle 外部验证 ────────────────────────────────────────────────────────
    kaggle_zs = kaggle_fs = None

    if kaggle_rows:
        print(f"\n[3/3] Kaggle 外部验证  (n={len(kaggle_rows)})...")

        # 5-shot 示例：从全部 UCI 中采样（Kaggle 无训练集）
        kaggle_examples = pick_examples(uci_rows, n=N_SHOT,
                                         seed=RANDOM_STATE)

        n_kag = min(len(kaggle_rows), QUICK) if QUICK else len(kaggle_rows)

        print(f"  Zero-shot ({n_kag} records)...")
        kaggle_zs = run_inference(kaggle_rows, examples=None,
                                  tag="kaggle_zs")
        print(f"  ZS  AUC={kaggle_zs['auc']:.4f}  "
              f"Sens={kaggle_zs['sensitivity']:.4f}  "
              f"Spec={kaggle_zs['specificity']:.4f}")

        print(f"  {N_SHOT}-shot ({n_kag} records)...")
        kaggle_fs = run_inference(kaggle_rows, examples=kaggle_examples,
                                  tag="kaggle_fs")
        print(f"  FS  AUC={kaggle_fs['auc']:.4f}  "
              f"Sens={kaggle_fs['sensitivity']:.4f}  "
              f"Spec={kaggle_fs['specificity']:.4f}")

    # ── 结果汇报 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    hdr = f"{'Metric':<14}  {'ZS-UCI (5fold)':<22}  {'5S-UCI (5fold)':<22}"
    if kaggle_zs:
        hdr += f"  {'ZS-Kaggle':<12}  {'5S-Kaggle':<12}"
    print(hdr)
    print("-" * (len(hdr) + 4))

    for m, label in [("auc", "AUC"),
                      ("sensitivity", "Sensitivity"),
                      ("specificity", "Specificity")]:
        zs_str = f"{zs_uci[m]['mean']:.4f}±{zs_uci[m]['std']:.4f}"
        fs_str = f"{fs_uci[m]['mean']:.4f}±{fs_uci[m]['std']:.4f}"
        line = f"  {label:<12}  {zs_str:<22}  {fs_str:<22}"
        if kaggle_zs:
            line += f"  {kaggle_zs[m]:.4f}        {kaggle_fs[m]:.4f}"
        print(line)

    # 核心问题：泛化差距
    if kaggle_zs:
        print("\n  ── 泛化差距 (Δ = Kaggle − UCI) ──")
        for tag, uci_s, kag_v in [
            ("ZS AUC",  zs_uci["auc"]["mean"],         kaggle_zs["auc"]),
            ("5S AUC",  fs_uci["auc"]["mean"],          kaggle_fs["auc"]),
            ("ZS Spec", zs_uci["specificity"]["mean"],  kaggle_zs["specificity"]),
            ("5S Spec", fs_uci["specificity"]["mean"],  kaggle_fs["specificity"]),
        ]:
            delta = kag_v - uci_s
            trend = "↓ collapse?" if delta < -0.10 else ("↑" if delta > 0 else "≈")
            print(f"    {tag:<10}: UCI={uci_s:.4f} → Kaggle={kag_v:.4f}  "
                  f"Δ={delta:+.4f}  {trend}")

    print("=" * 70)

    # ── 对比已有 Transformer 结果 ──────────────────────────────────────────────
    uci_json  = os.path.join(RESULTS_DIR, "results_kfold_5.json")
    kag_json  = os.path.join(RESULTS_DIR, "external_validation_results.json")
    if os.path.exists(uci_json) and os.path.exists(kag_json):
        with open(uci_json) as f:
            tf_uci = json.load(f)
        with open(kag_json) as f:
            tf_kag = json.load(f)

        # results_kfold_5.json: summary.val_auc / val_spec
        tf_uci_auc  = tf_uci["summary"]["val_auc"]["mean"]
        tf_uci_spec = tf_uci["summary"]["val_spec"]["mean"]
        # external_validation_results.json: summary.auc / specificity
        tf_kag_auc  = tf_kag["summary"]["auc"]["mean"]
        tf_kag_spec = tf_kag["summary"]["specificity"]["mean"]

        print("\n  ── 与 Custom Transformer 对比 ──")
        print(f"  {'Model':<24}  {'UCI-AUC':<14}  {'Kaggle-AUC':<12}  "
              f"{'UCI-Spec':<14}  Kaggle-Spec")
        print("  " + "-" * 76)
        print(f"  {'Custom Transformer':<24}  "
              f"{tf_uci_auc:.4f}          "
              f"{tf_kag_auc:.4f}        "
              f"{tf_uci_spec:.4f}          "
              f"{tf_kag_spec:.4f}")
        if kaggle_zs:
            print(f"  {'LLM Zero-shot':<24}  "
                  f"{zs_uci['auc']['mean']:.4f}±{zs_uci['auc']['std']:.4f}  "
                  f"{kaggle_zs['auc']:.4f}        "
                  f"{zs_uci['specificity']['mean']:.4f}±{zs_uci['specificity']['std']:.4f}  "
                  f"{kaggle_zs['specificity']:.4f}")
            print(f"  {'LLM 5-shot':<24}  "
                  f"{fs_uci['auc']['mean']:.4f}±{fs_uci['auc']['std']:.4f}  "
                  f"{kaggle_fs['auc']:.4f}        "
                  f"{fs_uci['specificity']['mean']:.4f}±{fs_uci['specificity']['std']:.4f}  "
                  f"{kaggle_fs['specificity']:.4f}")
        print("=" * 70)

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "model": MODEL,
        "experiment": f"zero-shot vs {N_SHOT}-shot",
        "random_state": RANDOM_STATE,
        "quick_mode": QUICK,
        "uci_zero_shot": {"fold_results": zs_folds, "summary": zs_uci},
        "uci_few_shot":  {"fold_results": fs_folds, "summary": fs_uci},
        "kaggle_zero_shot": kaggle_zs,
        "kaggle_few_shot":  kaggle_fs,
        "note": (
            "Kaggle ca/thal zero-padded (0) to match Transformer pipeline. "
            f"LLM sees ca=0 → 'Fluoroscopy vessels: 0' and "
            "thal=0 → 'Thalassemia: not available'."
        ),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存: {OUTPUT_JSON}")

    # 全部完成，删除检查点
    if os.path.exists(CKPT_JSON):
        os.remove(CKPT_JSON)
        print("✓ 检查点已清除")


if __name__ == "__main__":
    main()
