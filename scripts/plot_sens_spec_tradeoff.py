"""
scripts/plot_sens_spec_tradeoff.py

Sensitivity-Specificity 权衡面
每个点 = 一个模型在一个 fold 的 (Specificity, Sensitivity)
箭头连接同一模型在 UCI 的各 fold 均值，显示 N-shot 增加时的移动轨迹

Usage:
  uv run python scripts/plot_sens_spec_tradeoff.py
"""

import os, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── 加载数据 ───────────────────────────────────────────────────────────────────
llm = json.load(open(os.path.join(RESULTS, "experiment_ollama_results.json")))
sc  = json.load(open(os.path.join(RESULTS, "experiment_shotcurve_results.json")))
tf  = json.load(open(os.path.join(RESULTS, "results_kfold_5.json")))
kag = json.load(open(os.path.join(RESULTS, "external_validation_results.json")))
sc_kag = json.load(open(os.path.join(RESULTS, "experiment_shotcurve_results.json")))

# ── 数据组织 ───────────────────────────────────────────────────────────────────
# LLM shot → (sens_folds, spec_folds)
def llm_uci_folds(n):
    if n == 0:
        folds = llm["uci_zero_shot"]["fold_results"]
        return ([r["sensitivity"] for r in folds],
                [r["specificity"] for r in folds])
    elif n == 5:
        folds = llm["uci_few_shot"]["fold_results"]
        return ([r["sensitivity"] for r in folds],
                [r["specificity"] for r in folds])
    else:
        folds = sorted(sc[str(n)]["uci_folds"], key=lambda x: x["fold"])
        return ([r["sensitivity"] for r in folds],
                [r["specificity"] for r in folds])

def llm_kaggle(n):
    if n == 0:
        return llm["kaggle_zero_shot"]["sensitivity"], llm["kaggle_zero_shot"]["specificity"]
    elif n == 5:
        return llm["kaggle_few_shot"]["sensitivity"], llm["kaggle_few_shot"]["specificity"]
    else:
        k = sc_kag[str(n)]["kaggle"]
        return k["sensitivity"], k["specificity"]

# Transformer
tf_sens = [r["val_sens"] for r in tf["individual_results"]]
tf_spec = [r["val_spec"] for r in tf["individual_results"]]
tf_kag_sens = kag["summary"]["sensitivity"]["mean"]
tf_kag_spec = kag["summary"]["specificity"]["mean"]

SHOTS   = [0, 1, 3, 5, 10]
PALETTE = {
    0:  "#d73027",   # 红
    1:  "#fc8d59",   # 橙
    3:  "#fee090",   # 黄
    5:  "#91bfdb",   # 蓝
    10: "#4575b4",   # 深蓝
}
SHOT_LABEL = {0:"0-shot", 1:"1-shot", 3:"3-shot", 5:"5-shot", 10:"10-shot"}

# ══════════════════════════════════════════════════════════════════════════════
# Figure: 两图并排 — 左：UCI fold 散点，右：各模型均值 + Kaggle 对比
# ══════════════════════════════════════════════════════════════════════════════
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.suptitle("Sensitivity–Specificity Trade-off Space",
             fontsize=14, fontweight="bold", y=1.01)

# ── 左图：UCI 各 fold 散点 + 均值轨迹 ─────────────────────────────────────────
ax = axes[0]
ax.set_title("UCI 5-fold CV  (each dot = one fold)", fontweight="bold")

# 背景分区
ax.fill_between([0, 0.5], 0.5, 1.01, alpha=0.04, color="gray")
ax.fill_between([0.5, 1.01], 0.5, 1.01, alpha=0.06, color="green")
ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.text(0.52, 0.52, "Clinically\nuseful zone", fontsize=8.5,
        color="green", alpha=0.6)

# LLM 各 shot fold 散点
means_uci = {}   # n_shot → (mean_sens, mean_spec)
for n in SHOTS:
    sens_f, spec_f = llm_uci_folds(n)
    ax.scatter(spec_f, sens_f, color=PALETTE[n], alpha=0.45,
               s=55, zorder=3)
    ms, msp = float(np.mean(sens_f)), float(np.mean(spec_f))
    means_uci[n] = (ms, msp)
    ax.scatter(msp, ms, color=PALETTE[n], s=160, zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate(SHOT_LABEL[n],
                (msp, ms), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color=PALETTE[n], fontweight="bold")

# 均值轨迹箭头
for i in range(len(SHOTS)-1):
    n0, n1 = SHOTS[i], SHOTS[i+1]
    x0, y0 = means_uci[n0][1], means_uci[n0][0]
    x1, y1 = means_uci[n1][1], means_uci[n1][0]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.4, mutation_scale=12))

# Transformer UCI 均值
tf_m_sens = float(np.mean(tf_sens))
tf_m_spec = float(np.mean(tf_spec))
ax.scatter(tf_m_spec, tf_m_sens, color="#2c7bb6", s=200, marker="D",
           zorder=6, edgecolors="white", linewidths=1.5)
ax.annotate("Transformer", (tf_m_spec, tf_m_sens),
            textcoords="offset points", xytext=(6, -14),
            fontsize=9, color="#2c7bb6", fontweight="bold")
# Transformer fold scatter
ax.scatter(tf_spec, tf_sens, color="#2c7bb6", alpha=0.35, s=55, zorder=3)

ax.set_xlabel("Specificity (True Negative Rate)")
ax.set_ylabel("Sensitivity (True Positive Rate)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.15, 1.08)

# 图例
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=PALETTE[n],
           markersize=9, label=f"LLM {SHOT_LABEL[n]}")
    for n in SHOTS
] + [
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#2c7bb6",
           markersize=9, label="Custom Transformer"),
    Line2D([0],[0], color="#555555", lw=1.4,
           label="Trajectory (N-shot ↑)", marker=">", markersize=6),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5, framealpha=0.9)

# ── 右图：均值点 UCI vs Kaggle + 连线显示泛化位移 ─────────────────────────────
ax = axes[1]
ax.set_title("UCI Mean vs Kaggle  (arrows show generalization shift)",
             fontweight="bold")

ax.fill_between([0.5, 1.01], 0.5, 1.01, alpha=0.06, color="green")
ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.text(0.52, 0.52, "Clinically\nuseful zone", fontsize=8.5,
        color="green", alpha=0.6)

# LLM: UCI均值 → Kaggle，用箭头连接
for n in SHOTS:
    uci_s, uci_sp = means_uci[n]
    kag_s, kag_sp = llm_kaggle(n)

    # UCI 均值点（实心）
    ax.scatter(uci_sp, uci_s, color=PALETTE[n], s=160, zorder=5,
               edgecolors="white", linewidths=1.5)
    # Kaggle 点（空心）
    ax.scatter(kag_sp, kag_s, color=PALETTE[n], s=160, zorder=5,
               edgecolors=PALETTE[n], linewidths=2,
               facecolors="white")
    # 箭头：UCI → Kaggle
    ax.annotate("", xy=(kag_sp, kag_s), xytext=(uci_sp, uci_s),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE[n],
                                lw=1.5, linestyle="dashed",
                                mutation_scale=11))
    # 标签
    offset = (8, 4) if n != 1 else (8, -14)
    ax.annotate(SHOT_LABEL[n], (uci_sp, uci_s),
                textcoords="offset points", xytext=offset,
                fontsize=9, color=PALETTE[n], fontweight="bold")

# Transformer: UCI → Kaggle
ax.scatter(tf_m_spec, tf_m_sens, color="#2c7bb6", s=200, marker="D",
           zorder=6, edgecolors="white", linewidths=1.5)
ax.scatter(tf_kag_spec, tf_kag_sens, color="#2c7bb6", s=200, marker="D",
           zorder=6, edgecolors="#2c7bb6", linewidths=2, facecolors="white")
ax.annotate("", xy=(tf_kag_spec, tf_kag_sens),
            xytext=(tf_m_spec, tf_m_sens),
            arrowprops=dict(arrowstyle="-|>", color="#2c7bb6",
                            lw=2, linestyle="dashed", mutation_scale=13))
ax.annotate("Transformer", (tf_m_spec, tf_m_sens),
            textcoords="offset points", xytext=(-70, 6),
            fontsize=9, color="#2c7bb6", fontweight="bold")

ax.set_xlabel("Specificity (True Negative Rate)")
ax.set_ylabel("Sensitivity (True Positive Rate)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0.15, 1.08)

legend_elems2 = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=PALETTE[n],
           markersize=9, label=f"LLM {SHOT_LABEL[n]}")
    for n in SHOTS
] + [
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#2c7bb6",
           markersize=9, label="Custom Transformer"),
    mpatches.Patch(facecolor="white", edgecolor="gray",
                   label="Filled = UCI  |  Hollow = Kaggle"),
]
ax.legend(handles=legend_elems2, loc="lower left", fontsize=8.5, framealpha=0.9)

plt.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(FIGURES, f"sens_spec_tradeoff.{ext}")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  ✓ 保存: {path}")
plt.close()

print("\n完成。查看: open figures/sens_spec_tradeoff.png")
