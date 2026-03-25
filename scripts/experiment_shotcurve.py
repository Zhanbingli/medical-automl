"""
scripts/experiment_shotcurve.py

Few-shot 数量曲线实验：N-shot ∈ {1, 3, 10}
（0-shot 和 5-shot 已在 experiment_ollama_results.json 中，直接复用）

跑完后自动生成曲线图：figures/llm_shotcurve.pdf/png

Usage:
  uv run python scripts/experiment_shotcurve.py
"""

import os, sys, json, time, random, re
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── 配置 ──────────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
MODEL           = "qwen3.5:2b"
TEMPERATURE     = 0.0
MAX_TOKENS      = 512
WORKERS         = 1
REQUEST_TIMEOUT = 120
K_FOLDS         = 5
RANDOM_STATE    = 42

NEW_SHOTS = [1, 3, 10]   # 要新跑的；0 和 5 已有结果

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(ROOT, "patients.csv")
KAGGLE_CSV  = os.path.expanduser(
    "~/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
CKPT_JSON   = os.path.join(RESULTS_DIR, "shotcurve_checkpoint.json")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "experiment_shotcurve_results.json")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 特征解码 ──────────────────────────────────────────────────────────────────
SEX_DEC   = {0:"Female",1:"Male"}
CP_DEC    = {1:"typical angina",2:"atypical angina",3:"non-anginal pain",4:"asymptomatic"}
FBS_DEC   = {0:"<=120 mg/dl",1:">120 mg/dl"}
ECG_DEC   = {0:"normal",1:"ST-T wave abnormality",2:"left ventricular hypertrophy"}
EXANG_DEC = {0:"No",1:"Yes"}
SLP_DEC   = {0:"unknown",1:"upsloping",2:"flat",3:"downsloping"}
THAL_DEC  = {0:"not available",3:"normal",6:"fixed defect",7:"reversible defect"}

def format_patient(row):
    return (
        f"Age: {row['age']}, Sex: {SEX_DEC.get(row['sex'],row['sex'])}, "
        f"Chest pain: {CP_DEC.get(row['cp'],row['cp'])}, "
        f"Resting BP: {row['trestbps']} mmHg, Cholesterol: {row['chol']} mg/dl, "
        f"Fasting BS: {FBS_DEC.get(row['fbs'],row['fbs'])}, "
        f"Resting ECG: {ECG_DEC.get(row['restecg'],row['restecg'])}, "
        f"Max HR: {row['thalach']} bpm, "
        f"Exercise angina: {EXANG_DEC.get(row['exang'],row['exang'])}, "
        f"ST depression: {row['oldpeak']}, "
        f"ST slope: {SLP_DEC.get(row['slope'],row['slope'])}, "
        f"Vessels (ca): {row['ca']}, "
        f"Thalassemia: {THAL_DEC.get(row['thal'],row['thal'])}"
    )

_TASK = (
    "You are a clinical decision support AI. "
    "Assess the probability that the patient has heart disease.\n"
    "Output ONLY a JSON object, no other text: "
    '{"probability": <float 0.0-1.0>}'
)

def build_prompt(examples, patient_text):
    if not examples:
        return f"{_TASK}\n\nPatient: {patient_text}\n\nOutput:"
    ex_block = "\n\n".join(
        f"Example {i+1}: {ex['text']}\n"
        f"Diagnosis: {'heart disease present (1)' if ex['label']==1 else 'no heart disease (0)'}"
        for i, ex in enumerate(examples)
    )
    return (f"{_TASK}\n\nExamples:\n\n{ex_block}\n\n"
            f"Now assess: {patient_text}\n\nOutput:")

def call_ollama(prompt, retries=3):
    payload = {"model":MODEL,"prompt":prompt,"stream":False,
               "think":False,"keep_alive":-1,
               "options":{"temperature":TEMPERATURE,"num_predict":MAX_TOKENS}}
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            text = r.json().get("response","").strip()
            s, e = text.find("{"), text.rfind("}")+1
            if s >= 0 and e > s:
                try:
                    p = float(json.loads(text[s:e]).get("probability",-1))
                    if 0 <= p <= 1: return p
                except: pass
            for n in re.findall(r'\b(?:0\.\d+|1\.0+|0|1)\b', text):
                p = float(n)
                if 0 <= p <= 1: return p
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt < retries-1: time.sleep(2)
        except: pass
    return None

def pick_examples(rows, n, seed):
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"]==1]
    neg = [r for r in rows if r["label"]==0]
    chosen = rng.sample(pos, min(n//2, len(pos))) + rng.sample(neg, min(n-n//2, len(neg)))
    rng.shuffle(chosen)
    for ex in chosen: ex["text"] = format_patient(ex)
    return chosen

def _infer_one(args):
    idx, rec, prompt = args
    return idx, rec["label"], call_ollama(prompt)

def run_inference(records, examples, tag):
    tasks = [(i, rec, build_prompt(examples, format_patient(rec)))
             for i, rec in enumerate(records)]
    slot = [None]*len(records)
    n_failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fut_map = {pool.submit(_infer_one, t): t[0] for t in tasks}
        with tqdm(total=len(records), desc=f"  {tag}", unit="rec") as pbar:
            for fut in as_completed(fut_map):
                idx, label, prob = fut.result()
                if prob is None: prob = 0.5; n_failed += 1
                slot[idx] = (label, prob)
                pbar.update(1)
                pbar.set_postfix(failed=n_failed)

    y_true = [s[0] for s in slot]
    y_prob = [s[1] for s in slot]
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    auc  = roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "auc":         float(auc),
        "sensitivity": float(tp/(tp+fn)) if (tp+fn) > 0 else 0.0,
        "specificity": float(tn/(tn+fp)) if (tn+fp) > 0 else 0.0,
        "n_failed":    n_failed,
        "elapsed_s":   round(time.time()-t0, 1),
    }

def summarize(fold_results):
    out = {}
    for m in ["auc","sensitivity","specificity"]:
        vals = [r[m] for r in fold_results if not np.isnan(r[m])]
        out[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out

def _safe_int(v, default=0):
    try: return int(float(v))
    except: return default

def load_uci():
    df = pd.read_csv(CSV_PATH)
    df["num_binary"] = df["num"].apply(lambda x: 0 if x==0 else 1)
    return [{"age":int(r["age"]),"sex":int(r["sex"]),"cp":int(r["cp"]),
             "trestbps":int(r["trestbps"]),"chol":int(r["chol"]),"fbs":int(r["fbs"]),
             "restecg":int(r["restecg"]),"thalach":int(r["thalach"]),
             "exang":int(r["exang"]),"oldpeak":float(r["oldpeak"]),
             "slope":int(r["slope"]),"ca":_safe_int(r["ca"]),
             "thal":_safe_int(r["thal"]),"label":int(r["num_binary"])}
            for _,r in df.iterrows()]

def load_kaggle():
    SEX={"M":1,"F":0}; CP={"TA":1,"ATA":2,"NAP":3,"ASY":4}
    ECG={"Normal":0,"ST":1,"LVH":2}; ANG={"Y":1,"N":0}; SLP={"Up":1,"Flat":2,"Down":3}
    df = pd.read_csv(KAGGLE_CSV)
    return [{"age":int(r["Age"]),"sex":SEX.get(str(r["Sex"]),0),
             "cp":CP.get(str(r["ChestPainType"]),0),"trestbps":int(r["RestingBP"]),
             "chol":int(r["Cholesterol"]),"fbs":int(r["FastingBS"]),
             "restecg":ECG.get(str(r["RestingECG"]),0),"thalach":int(r["MaxHR"]),
             "exang":ANG.get(str(r["ExerciseAngina"]),0),"oldpeak":float(r["Oldpeak"]),
             "slope":SLP.get(str(r["ST_Slope"]),0),"ca":0,"thal":0,
             "label":int(r["HeartDisease"])}
            for _,r in df.iterrows()]

# ── 检查点 ────────────────────────────────────────────────────────────────────
def load_ckpt():
    if os.path.exists(CKPT_JSON):
        with open(CKPT_JSON) as f: return json.load(f)
    return {}

def save_ckpt(data):
    with open(CKPT_JSON, "w") as f: json.dump(data, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("="*70)
    print(f"Few-shot Curve Experiment  N-shot ∈ {NEW_SHOTS}")
    print(f"Model: {MODEL}")
    print("="*70)

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] 无法连接 Ollama: {e}\n  请先运行: ollama serve"); sys.exit(1)

    uci_rows    = load_uci()
    kaggle_rows = load_kaggle() if os.path.exists(KAGGLE_CSV) else None
    labels_arr  = np.array([r["label"] for r in uci_rows])
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds_list  = list(skf.split(uci_rows, labels_arr))

    ckpt = load_ckpt()

    for n_shot in NEW_SHOTS:
        key = str(n_shot)
        if key not in ckpt:
            ckpt[key] = {"uci_folds": [], "kaggle": None}

        done_folds = {r["fold"] for r in ckpt[key]["uci_folds"]}
        print(f"\n{'─'*70}")
        print(f"  {n_shot}-shot  已完成 fold: {sorted(done_folds)}")

        # UCI 5-fold CV
        for fold_idx, (train_idx, val_idx) in enumerate(folds_list):
            if fold_idx in done_folds: continue
            train_rows = [uci_rows[i] for i in train_idx]
            val_rows   = [uci_rows[i] for i in val_idx]
            examples   = pick_examples(train_rows, n_shot, RANDOM_STATE+fold_idx)

            print(f"\n  {n_shot}-shot  Fold {fold_idx}  (val={len(val_rows)})")
            res = run_inference(val_rows, examples, f"{n_shot}s_fold{fold_idx}")
            res["fold"] = fold_idx
            ckpt[key]["uci_folds"].append(res)
            save_ckpt(ckpt)
            print(f"  AUC={res['auc']:.4f}  Sens={res['sensitivity']:.4f}  "
                  f"Spec={res['specificity']:.4f}  failed={res['n_failed']}")

        # Kaggle 外部验证
        if kaggle_rows and ckpt[key]["kaggle"] is None:
            examples = pick_examples(uci_rows, n_shot, RANDOM_STATE)
            print(f"\n  {n_shot}-shot  Kaggle (n={len(kaggle_rows)})")
            res = run_inference(kaggle_rows, examples, f"{n_shot}s_kaggle")
            ckpt[key]["kaggle"] = res
            save_ckpt(ckpt)
            print(f"  AUC={res['auc']:.4f}  Sens={res['sensitivity']:.4f}  "
                  f"Spec={res['specificity']:.4f}")

    # ── 汇总并保存 JSON ────────────────────────────────────────────────────────
    output = {}
    for n_shot in NEW_SHOTS:
        key = str(n_shot)
        output[key] = {
            "uci_summary": summarize(ckpt[key]["uci_folds"]),
            "uci_folds":   ckpt[key]["uci_folds"],
            "kaggle":      ckpt[key]["kaggle"],
        }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ 新实验结果已保存: {OUTPUT_JSON}")

    # 清理检查点
    os.remove(CKPT_JSON)

    # ── 绘图 ───────────────────────────────────────────────────────────────────
    plot_shotcurve(output)


def plot_shotcurve(new_results):
    """合并已有 0-shot/5-shot 结果，画出 N-shot vs Metrics 曲线。"""

    # 读取已有 0-shot 和 5-shot
    with open(os.path.join(RESULTS_DIR, "experiment_ollama_results.json")) as f:
        existing = json.load(f)

    # 构建完整数据点 {n_shot: {metric: {uci_mean, uci_std, kaggle}}}
    def e2pt(uci_summary, kaggle_result):
        return {m: {"uci": uci_summary[m]["mean"],
                    "uci_std": uci_summary[m]["std"],
                    "kaggle": kaggle_result[m]}
                for m in ["auc", "sensitivity", "specificity"]}

    all_data = {}
    all_data[0]  = e2pt(existing["uci_zero_shot"]["summary"],
                        existing["kaggle_zero_shot"])
    all_data[5]  = e2pt(existing["uci_few_shot"]["summary"],
                        existing["kaggle_few_shot"])
    for n in [1, 3, 10]:
        d = new_results[str(n)]
        all_data[n] = e2pt(d["uci_summary"], d["kaggle"])

    shots = sorted(all_data.keys())   # [0, 1, 3, 5, 10]

    # ── 画图 ──────────────────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.family":"Times New Roman","font.size":12})

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Few-shot Count vs Performance: LLM on UCI (Internal) and Kaggle (External)",
                 fontsize=13, fontweight="bold", y=1.01)

    metric_labels = {"auc":"AUC","sensitivity":"Sensitivity","specificity":"Specificity"}
    col_uci = "#2166ac"
    col_kag = "#d6604d"

    for ax, metric in zip(axes, ["auc","sensitivity","specificity"]):
        uci_vals = [all_data[n][metric]["uci"]     for n in shots]
        uci_errs = [all_data[n][metric]["uci_std"] for n in shots]
        kag_vals = [all_data[n][metric]["kaggle"]  for n in shots]

        ax.fill_between(shots,
                        [v-e for v,e in zip(uci_vals, uci_errs)],
                        [v+e for v,e in zip(uci_vals, uci_errs)],
                        alpha=0.15, color=col_uci)
        ax.plot(shots, uci_vals, "o-", color=col_uci, linewidth=2,
                markersize=7, label="UCI (Internal CV)")
        ax.plot(shots, kag_vals, "s--", color=col_kag, linewidth=2,
                markersize=7, label="Kaggle (External)")

        # 数值标注
        for n, uv, kv in zip(shots, uci_vals, kag_vals):
            ax.annotate(f"{uv:.2f}", (n, uv), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, color=col_uci)
            ax.annotate(f"{kv:.2f}", (n, kv), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8.5, color=col_kag)

        # Specificity 图加临床参考线
        if metric == "specificity":
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                       label="Clinically marginal (0.5)")
            ax.fill_between([0,10], 0.5, 1.0, alpha=0.04, color="green")
            ax.text(9.5, 0.52, "clinically\nuseful", ha="right",
                    fontsize=8, color="green", alpha=0.7)

        ax.set_xlabel("Number of Few-shot Examples")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(metric_labels[metric], fontweight="bold")
        ax.set_xticks(shots)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf","png"):
        path = os.path.join(FIGURES_DIR, f"llm_shotcurve.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ 保存: {path}")
    plt.close()
    print("\n完成。查看图片: open figures/llm_shotcurve.png")


if __name__ == "__main__":
    main()
