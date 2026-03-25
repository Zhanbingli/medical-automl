"""
scripts/plot_llm_comparison.py

可视化 LLM 实验结果（experiment_ollama_results.json）
生成两张图：
  Figure A: 泛化差距对比（核心结果）—— UCI vs Kaggle，AUC 和 Specificity
  Figure B: 全指标对比（四个模型 × 四个指标）

Usage:
  uv run python scripts/plot_llm_comparison.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── 路径 ───────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS   = os.path.join(ROOT, "results")
FIGURES   = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── 样式 ───────────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

COLORS = {
    "transformer": "#2c7bb6",
    "llm_zs":      "#d7191c",
    "llm_fs":      "#1a9641",
}

# ── 读取数据 ───────────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "experiment_ollama_results.json")) as f:
    llm = json.load(f)

with open(os.path.join(RESULTS, "results_kfold_5.json")) as f:
    tf_uci_raw = json.load(f)

with open(os.path.join(RESULTS, "external_validation_results.json")) as f:
    tf_kag_raw = json.load(f)

# Custom Transformer 数据
tf = {
    "uci_auc":  tf_uci_raw["summary"]["val_auc"]["mean"],
    "uci_sens": tf_uci_raw["summary"]["val_sens"]["mean"],
    "uci_spec": tf_uci_raw["summary"]["val_spec"]["mean"],
    "uci_acc":  tf_uci_raw["summary"]["val_acc"]["mean"],
    "uci_auc_std":  tf_uci_raw["summary"]["val_auc"]["std"],
    "uci_spec_std": tf_uci_raw["summary"]["val_spec"]["std"],
    "kag_auc":  tf_kag_raw["summary"]["auc"]["mean"],
    "kag_sens": tf_kag_raw["summary"]["sensitivity"]["mean"],
    "kag_spec": tf_kag_raw["summary"]["specificity"]["mean"],
    "kag_acc":  tf_kag_raw["summary"]["acc"]["mean"],
    "kag_auc_std":  tf_kag_raw["summary"]["auc"]["std"],
    "kag_spec_std": tf_kag_raw["summary"]["specificity"]["std"],
}

# LLM 数据
zs_uci = llm["uci_zero_shot"]["summary"]
fs_uci = llm["uci_few_shot"]["summary"]
zs_kag = llm["kaggle_zero_shot"]
fs_kag = llm["kaggle_few_shot"]

# ══════════════════════════════════════════════════════════════════════════════
# Figure A：泛化差距对比（AUC + Specificity，UCI vs Kaggle）
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Generalization Gap: Internal (UCI) vs External (Kaggle) Validation",
             fontsize=14, fontweight="bold", y=1.01)

models     = ["Custom\nTransformer", "LLM\nZero-shot", "LLM\n5-shot"]
model_keys = ["transformer", "llm_zs", "llm_fs"]
col_uci    = "#4393c3"
col_kag    = "#d6604d"

for ax, (metric, title, ylabel) in zip(axes, [
    ("auc",  "AUC Comparison",         "AUC (Area Under ROC Curve)"),
    ("spec", "Specificity Comparison",  "Specificity (True Negative Rate)"),
]):
    uci_vals = [
        tf[f"uci_{metric}"],
        zs_uci[metric]["mean"] if metric != "spec" else zs_uci["specificity"]["mean"],
        fs_uci[metric]["mean"] if metric != "spec" else fs_uci["specificity"]["mean"],
    ]
    uci_errs = [
        tf[f"uci_{metric}_std"],
        zs_uci[metric]["std"] if metric != "spec" else zs_uci["specificity"]["std"],
        fs_uci[metric]["std"] if metric != "spec" else fs_uci["specificity"]["std"],
    ]
    kag_vals = [
        tf[f"kag_{metric}"],
        zs_kag[metric] if metric != "spec" else zs_kag["specificity"],
        fs_kag[metric] if metric != "spec" else fs_kag["specificity"],
    ]
    kag_errs = [
        tf[f"kag_{metric}_std"],
        0, 0,  # Kaggle LLM 只跑一次，无 std
    ]

    x      = np.arange(len(models))
    width  = 0.35

    bars1 = ax.bar(x - width/2, uci_vals, width, yerr=uci_errs,
                   color=col_uci, alpha=0.85, capsize=4,
                   label="UCI (Internal CV)", error_kw={"linewidth": 1.2})
    bars2 = ax.bar(x + width/2, kag_vals, width, yerr=kag_errs,
                   color=col_kag, alpha=0.85, capsize=4,
                   label="Kaggle (External)", error_kw={"linewidth": 1.2})

    # 数值标注
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # 箭头标注 Transformer 的 collapse
    if metric == "spec":
        tf_uci_v = uci_vals[0]
        tf_kag_v = kag_vals[0]
        ax.annotate(
            f"Δ={tf_kag_v - tf_uci_v:+.3f}\n(Collapse)",
            xy=(x[0] + width/2, tf_kag_v),
            xytext=(x[0] + width/2 + 0.45, tf_kag_v + 0.08),
            fontsize=9, color="#b2182b", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.5),
        )
        # LLM 5-shot 稳定标注
        fs_uci_v = uci_vals[2]
        fs_kag_v = kag_vals[2]
        ax.annotate(
            f"Δ={fs_kag_v - fs_uci_v:+.3f}\n(Stable)",
            xy=(x[2] + width/2, fs_kag_v),
            xytext=(x[2] + width/2 + 0.35, fs_kag_v + 0.10),
            fontsize=9, color="#1a9850", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a9850", lw=1.5),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(FIGURES, f"llm_generalization_gap.{ext}")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 保存: {path}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure B：全指标对比热力表（4 模型 × UCI/Kaggle × 4 指标）
# ══════════════════════════════════════════════════════════════════════════════
rows = [
    ("Custom Transformer", "UCI",
     tf["uci_auc"], tf["uci_sens"], tf["uci_spec"], tf["uci_acc"]),
    ("Custom Transformer", "Kaggle",
     tf["kag_auc"], tf["kag_sens"], tf["kag_spec"], tf["kag_acc"]),
    ("LLM Zero-shot",      "UCI",
     zs_uci["auc"]["mean"], zs_uci["sensitivity"]["mean"],
     zs_uci["specificity"]["mean"], zs_uci["acc"]["mean"]),
    ("LLM Zero-shot",      "Kaggle",
     zs_kag["auc"], zs_kag["sensitivity"], zs_kag["specificity"], zs_kag["acc"]),
    ("LLM 5-shot",         "UCI",
     fs_uci["auc"]["mean"], fs_uci["sensitivity"]["mean"],
     fs_uci["specificity"]["mean"], fs_uci["acc"]["mean"]),
    ("LLM 5-shot",         "Kaggle",
     fs_kag["auc"], fs_kag["sensitivity"], fs_kag["specificity"], fs_kag["acc"]),
]

col_labels = ["AUC", "Sensitivity", "Specificity", "Accuracy"]
data_matrix = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
row_labels  = [f"{r[0]}\n({r[1]})" for r in rows]

fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Full Metrics Comparison: Custom Transformer vs LLM",
             fontsize=13, fontweight="bold")

im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0.3, vmax=0.95, aspect="auto")

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontweight="bold")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)

# 数值标注
for i in range(len(rows)):
    for j in range(len(col_labels)):
        val = data_matrix[i, j]
        color = "black" if 0.4 < val < 0.8 else "white"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

# 横线分隔三个模型
for y in [1.5, 3.5]:
    ax.axhline(y, color="white", linewidth=2.5)

plt.colorbar(im, ax=ax, label="Score", shrink=0.8)
plt.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(FIGURES, f"llm_full_metrics.{ext}")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 保存: {path}")
plt.close()

print("\n完成。两张图已保存到 figures/")
print("  llm_generalization_gap.pdf/png  ← 核心结果图")
print("  llm_full_metrics.pdf/png        ← 全指标热力图")
