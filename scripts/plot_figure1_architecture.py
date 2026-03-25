"""
scripts/plot_figure1_architecture.py

Figure 1: System Architecture — Tabular-to-Text Pipeline, Model Architectures, and Evaluation

Three-column layout:
  Left:   Tabular-to-Text Encoding (shared input pipeline)
  Center: Two model pathways (Custom Transformer vs LLM)
  Right:  Evaluation Framework (internal CV + external validation)

Usage:
    uv run python scripts/plot_figure1_architecture.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

# ── Color palette ─────────────────────────────────────────────────────────────
C_DATA    = "#e8f4f8"  # light blue - data
C_PIPE    = "#d4e8d4"  # light green - pipeline
C_TRANS   = "#fff2cc"  # light yellow - transformer
C_LLM     = "#fde0d0"  # light peach - LLM
C_EVAL    = "#e8e0f0"  # light purple - evaluation
C_BORDER  = "#555555"
C_ARROW   = "#333333"
C_ACCENT  = "#2c7bb6"  # blue accent
C_RED     = "#d73027"  # red accent


def add_box(ax, x, y, w, h, text, facecolor, fontsize=9, bold=False,
            edgecolor=C_BORDER, linewidth=1.2, text_color="black"):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, zorder=2
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=text_color, zorder=3)
    return box


def add_arrow(ax, x1, y1, x2, y2, color=C_ARROW, style="-|>", lw=1.5):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color,
        linewidth=lw, mutation_scale=14, zorder=4
    )
    ax.add_patch(arrow)


def add_section_label(ax, x, y, text, fontsize=12):
    """Add a bold section label."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#333333", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN A: Tabular-to-Text Encoding (x: 0.3 – 4.5)
# ══════════════════════════════════════════════════════════════════════════════
add_section_label(ax, 2.4, 8.5, "A. Tabular-to-Text Encoding")

# Data sources
add_box(ax, 0.5, 7.2, 3.8, 0.9,
        "UCI Heart Disease Dataset\nn = 303, 13 features",
        C_DATA, fontsize=9, bold=True)
add_box(ax, 0.5, 5.9, 3.8, 0.9,
        "Kaggle Heart Disease Dataset\nn = 918, 13 features (zero-padded ca/thal)",
        C_DATA, fontsize=8.5, bold=True)

add_arrow(ax, 2.4, 5.9, 2.4, 5.35)

# Encoding process
add_box(ax, 0.5, 4.3, 3.8, 0.95,
        "Structured → English Narrative\n\"Age 58, Sex Male, Chest Pain Type 4,\n"
        "Resting BP 130 mmHg, ...\"",
        C_PIPE, fontsize=8)

add_arrow(ax, 2.4, 7.2, 2.4, 5.35)
add_arrow(ax, 2.4, 4.3, 2.4, 3.65)

# Shared text output
add_box(ax, 0.5, 2.6, 3.8, 0.95,
        "Standardized Clinical Narrative\n(identical text input for both models)",
        C_PIPE, fontsize=8.5, bold=True)

# Split arrow to two pathways
add_arrow(ax, 4.3, 3.08, 5.5, 6.55, color=C_ACCENT, lw=1.8)
add_arrow(ax, 4.3, 3.08, 5.5, 2.75, color=C_RED, lw=1.8)

# Labels on arrows
ax.text(4.6, 5.3, "BPE\ntokenization", fontsize=7.5, color=C_ACCENT,
        ha="center", va="center", fontstyle="italic", rotation=50)
ax.text(4.6, 2.5, "Natural language\nprompt", fontsize=7.5, color=C_RED,
        ha="center", va="center", fontstyle="italic", rotation=-10)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN B-top: Custom Transformer (x: 5.5 – 10.0)
# ══════════════════════════════════════════════════════════════════════════════
add_section_label(ax, 7.75, 8.5, "B. Model Architectures")

# Transformer box (larger, with internal details)
tf_box = FancyBboxPatch(
    (5.5, 5.0), 4.5, 3.2,
    boxstyle="round,pad=0.2",
    facecolor=C_TRANS, edgecolor=C_ACCENT,
    linewidth=2.0, linestyle="-", zorder=1
)
ax.add_patch(tf_box)
ax.text(7.75, 7.85, "Custom Tabular-to-Text Transformer",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=C_ACCENT, zorder=3)

add_box(ax, 5.9, 7.0, 3.7, 0.6,
        "BPE Tokenizer (vocab = 8,192)", "#fffbe6", fontsize=8.5)
add_arrow(ax, 7.75, 7.0, 7.75, 6.65)
add_box(ax, 5.9, 5.95, 3.7, 0.6,
        "GPT Decoder-only + Rotary PE\nValue Embedding Enhancement",
        "#fffbe6", fontsize=8)
add_arrow(ax, 7.75, 5.95, 7.75, 5.6)
add_box(ax, 5.9, 5.0, 3.7, 0.55,
        "5-fold Stratified CV Training", "#fffbe6", fontsize=8.5)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN B-bottom: LLM (x: 5.5 – 10.0)
# ══════════════════════════════════════════════════════════════════════════════
llm_box = FancyBboxPatch(
    (5.5, 0.8), 4.5, 3.2,
    boxstyle="round,pad=0.2",
    facecolor=C_LLM, edgecolor=C_RED,
    linewidth=2.0, linestyle="-", zorder=1
)
ax.add_patch(llm_box)
ax.text(7.75, 3.65, "LLM (Qwen3.5-2B, Local Deployment)",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=C_RED, zorder=3)

add_box(ax, 5.9, 2.7, 3.7, 0.65,
        "In-context Learning\n0 / 1 / 3 / 5 / 10 shot examples",
        "#fff0e6", fontsize=8.5)
add_arrow(ax, 7.75, 2.7, 7.75, 2.35)
add_box(ax, 5.9, 1.7, 3.7, 0.6,
        "Prompt: narrative + task + examples\nOutput: {\"probability\": float}",
        "#fff0e6", fontsize=8)
add_arrow(ax, 7.75, 1.7, 7.75, 1.35)
add_box(ax, 5.9, 0.85, 3.7, 0.5,
        "No weight updates (frozen)", "#fff0e6", fontsize=8.5)

# Key difference annotation
ax.annotate(
    "Key difference:\nTransformer learns task-specific\nweights; LLM uses frozen\npre-trained knowledge",
    xy=(5.3, 4.5), fontsize=7.5, fontstyle="italic", color="#666666",
    ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#cccccc", linewidth=0.8)
)


# ══════════════════════════════════════════════════════════════════════════════
# COLUMN C: Evaluation Framework (x: 10.5 – 15.5)
# ══════════════════════════════════════════════════════════════════════════════
add_section_label(ax, 13.0, 8.5, "C. Evaluation Framework")

# Arrows from models to evaluation
add_arrow(ax, 10.0, 6.5, 11.0, 7.1, color=C_ACCENT, lw=1.5)
add_arrow(ax, 10.0, 2.5, 11.0, 7.1, color=C_RED, lw=1.5)

# Internal validation
add_box(ax, 11.0, 6.6, 4.3, 1.2,
        "Internal Validation\nUCI 5-Fold Stratified CV (n = 303)\nMetrics: AUC, Sensitivity, Specificity, Accuracy\nPaired t-test + Cohen's d",
        C_EVAL, fontsize=8.5, bold=False)

add_arrow(ax, 13.15, 6.6, 13.15, 6.05)

# External validation
add_box(ax, 11.0, 4.85, 4.3, 1.1,
        "External Validation\nKaggle Cohort (n = 918)\nca, thal zero-padded → OOD tokens\nGeneralization Gap: Δ = External − Internal",
        C_EVAL, fontsize=8.5)

add_arrow(ax, 13.15, 4.85, 13.15, 4.3)

# Key outputs
add_box(ax, 11.0, 3.1, 4.3, 1.1,
        "Mechanistic Analysis\n• Attention weight extraction (last layer)\n"
        "• OOD comparison: UCI-style vs Kaggle-style\n"
        "• Imputation strategy ablation",
        C_EVAL, fontsize=8.5)

add_arrow(ax, 13.15, 3.1, 13.15, 2.55)

# Core findings
findings_box = FancyBboxPatch(
    (11.0, 0.7), 4.3, 1.75,
    boxstyle="round,pad=0.2",
    facecolor="#f0f0f0", edgecolor="#333333",
    linewidth=2.0, zorder=1
)
ax.add_patch(findings_box)
ax.text(13.15, 2.15, "Core Findings", ha="center", va="center",
        fontsize=10, fontweight="bold", color="#333333", zorder=3)
ax.text(13.15, 1.1,
        "Transformer: Specificity Collapse\n"
        "(Δ Spec = −0.156, OOD attention disruption)\n\n"
        "LLM 5-shot: Stable generalization\n"
        "(Δ Spec = +0.041, calibration drift corrected)",
        ha="center", va="center", fontsize=8, color="#333333", zorder=3)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
for ext in ("pdf", "png"):
    path = os.path.join(FIGURES, f"figure1_architecture.{ext}")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  ✓ Saved: {path}")
plt.close()

print("\nDone. View: open figures/figure1_architecture.png")
