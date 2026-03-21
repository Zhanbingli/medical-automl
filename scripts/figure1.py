import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import warnings

# --- Global SCI Font & Style Configuration ------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
warnings.simplefilter("ignore", UserWarning)

# --- Color Palette (Grayscale Academic Style) ──────────────────────────────────
CLR_BACK = "#FFFFFF"
CLR_LINE = "#000000"
CLR_TEXT = "#000000"
CLR_NODE_BG = "#FFFFFF"
CLR_NODE_EMP = "#F0F0F0"
CLR_MODULE_EDGE = "#555555"

# --- Helper Functions ─────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, facecolor=CLR_NODE_BG, edgecolor=CLR_LINE, lw=1.2, radius=0.06, zorder=2):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

def label(ax, x, y, text, size=8, bold=False, color=CLR_TEXT, zorder=5, italic=False, rotation=0, ha="center"):
    weight = "bold" if bold else "normal"
    style = "italic" if italic else "normal"
    ax.text(x, y, text, ha=ha, va="center", rotation=rotation,
            fontsize=size, fontweight=weight, fontstyle=style, color=color, zorder=zorder)

def arrow(ax, x1, y1, x2, y2, color=CLR_LINE, lw=1.2, style="-|>", zorder=3, ls="-"):
    # 纯手动直接连线，绝对不允许库自动生成弧线或乱折线
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls), zorder=zorder)

def module_bg(ax, x, y, w, h, title, zorder=1):
    bg = FancyBboxPatch((x, y), w, h,
                        boxstyle="square,pad=0.0",
                        facecolor="none", edgecolor=CLR_MODULE_EDGE,
                        linewidth=1.2, linestyle="--", zorder=zorder)
    ax.add_patch(bg)
    # 精确左上对齐，防止标题溢出虚线边界
    ax.text(x + 0.15, y + h - 0.15, title, ha="left", va="top",
            fontsize=10, fontweight="bold", color=CLR_TEXT, zorder=zorder+1)

# --- Canvas Initialization ----------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 8.5))
fig.patch.set_facecolor(CLR_BACK)
ax.set_facecolor(CLR_BACK)
ax.set_xlim(0, 16)
ax.set_ylim(0, 8.5)
ax.set_aspect("equal")
ax.axis("off")

# ══════════════════════════════════════════════════════════════════════════════
# MODULE A  –  Tabular-to-Text Pipeline
# ══════════════════════════════════════════════════════════════════════════════
module_bg(ax, 0.25, 0.4, 4.50, 7.5, "A. Tabular-to-Text Pipeline")

rbox(ax, 2.50, 7.05, 3.6, 0.80)
label(ax, 2.50, 7.20, "UCI Heart Disease Dataset", size=8.5, bold=True)
label(ax, 2.50, 6.90, "n = 303 records · 13 clinical features", size=7.5)

arrow(ax, 2.50, 6.65, 2.50, 5.95)
label(ax, 2.65, 6.30, "template fill", size=7.5, italic=True, color="#555555", ha="left")

rbox(ax, 2.50, 5.60, 3.6, 0.65)
label(ax, 2.50, 5.75, "Chinese Structured Narrative", size=8.5, bold=True)
label(ax, 2.50, 5.48, '"Patient: age=63, sex=male, cp=1 ..."', size=7.5, italic=True)

arrow(ax, 2.50, 5.28, 2.50, 4.60)
label(ax, 2.65, 4.95, "BPE tokenizer (vocab=8,192)", size=7.5, italic=True, color="#555555", ha="left")

rbox(ax, 2.50, 4.30, 3.6, 0.55, facecolor=CLR_NODE_EMP)
label(ax, 2.50, 4.44, "Token Sequence", size=8.5, bold=True)
label(ax, 2.50, 4.17, r"$[BOS]$ $t_1$ $t_2$ $\dots$ $t_{n-1} \rightarrow \text{predict } t_n$", size=7.5)

for i in range(5):
    rbox(ax, 1.05 + i*0.55, 3.70, 0.46, 0.30, facecolor="#F5F5F5", lw=0.8, radius=0.03)
    label(ax, 1.05 + i*0.55, 3.70, f"$t_{{{i+1}}}$", size=7)
label(ax, 4.10, 3.70, "…", size=10)

label(ax, 0.65, 3.35, "OUTPUT", size=7.0, bold=True, color="#888888", rotation=90)

# A -> B 走线 (避开模型框的直接贯穿，采用 L 型电路走线)
arrow(ax, 4.30, 4.30, 4.70, 4.30, style="-")
arrow(ax, 4.70, 4.30, 4.70, 5.55, style="-")
arrow(ax, 4.70, 5.55, 5.65, 5.55, style="-|>")
label(ax, 4.55, 4.92, "Token input", size=7.5, bold=True, rotation=90)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE B  –  Agent-Driven NAS Loop
# ══════════════════════════════════════════════════════════════════════════════
module_bg(ax, 5.00, 0.4, 5.50, 7.5, "B. Agent-Driven NAS Loop")

rbox(ax, 7.75, 7.10, 4.0, 0.85, facecolor=CLR_NODE_EMP)
label(ax, 7.75, 7.30, "LLM Agent", size=9.0, bold=True)
label(ax, 7.75, 7.03, "OpenCode · Kimi-k2.5 · 256 K context", size=7.5)

arrow(ax, 7.75, 6.67, 7.75, 5.95)
label(ax, 7.90, 6.31, "edit train.py", size=7.5, italic=True, color="#555555", ha="left")

rbox(ax, 7.75, 5.55, 4.2, 0.80)
label(ax, 7.75, 5.75, "Decoder-only Transformer", size=8.5, bold=True)
label(ax, 7.75, 5.45, "Search space: Depth · Aspect Ratio · Dropout", size=7.5)

params = [("Depth", "3"), ("AR", "48"), ("Drop", "0.2")]
for i, (p, v) in enumerate(params):
    cx = 6.60 + i*1.20
    rbox(ax, cx, 5.05, 0.90, 0.28, facecolor="#EEEEEE", lw=0.8, radius=0.04)
    label(ax, cx, 5.05, f"{p}={v}", size=7.5)

label(ax, 10.15, 5.58, "[5 min / iter]", size=7.5, italic=True, color="#666666")

arrow(ax, 7.75, 5.15, 7.75, 4.00)
label(ax, 7.90, 4.57, "val AUC", size=7.5, italic=True, color="#555555", ha="left")

rbox(ax, 7.75, 3.60, 4.0, 0.80, facecolor=CLR_NODE_EMP)
label(ax, 7.75, 3.80, "Reward + Rollback", size=8.5, bold=True)
label(ax, 7.75, 3.52, "AUC reward signal → keep or discard", size=7.5)

rbox(ax, 7.75, 1.50, 4.0, 0.65, facecolor="#F5F5F5")
label(ax, 7.75, 1.68, "Converged Configuration", size=8.5, bold=True)
label(ax, 7.75, 1.42, "Depth=3 · AR=48 · Dropout=0.2", size=7.5)

# KEEP / DISCARD (纯净的直角正交分支)
arrow(ax, 7.75, 3.20, 7.75, 1.83) # 贯穿的主轴
arrow(ax, 7.75, 2.95, 6.00, 2.95, style="-")
arrow(ax, 6.00, 2.95, 6.00, 2.75)
rbox(ax, 6.00, 2.50, 1.6, 0.50, facecolor="#F5F5F5")
label(ax, 6.00, 2.62, "[+] AUC UP", size=7.5, bold=True)
label(ax, 6.00, 2.38, "git commit (KEEP)", size=6.5)

arrow(ax, 7.75, 2.95, 9.50, 2.95, style="-")
arrow(ax, 9.50, 2.95, 9.50, 2.75)
rbox(ax, 9.50, 2.50, 1.6, 0.50, facecolor="#F5F5F5")
label(ax, 9.50, 2.62, "[-] AUC DOWN", size=7.5, bold=True)
label(ax, 9.50, 2.38, "git reset (DISCARD)", size=6.5)

# Feedback Loop
loop_x = 5.25
arrow(ax, 5.75, 3.60, loop_x, 3.60, lw=1.5, ls="--", style="-")
arrow(ax, loop_x, 3.60, loop_x, 7.10, lw=1.5, ls="--", style="-")
arrow(ax, loop_x, 7.10, 5.75, 7.10, lw=1.5, ls="--")
label(ax, loop_x - 0.15, 5.35, "AUC Reward Feedback", size=8.0, bold=True, rotation=90)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE C – Evaluation (教科书级对称分布与电路级走线)
# ══════════════════════════════════════════════════════════════════════════════
module_bg(ax, 10.75, 0.4, 5.00, 7.5, "C. Evaluation")

# 1. 顶部数据总线 (Best Model 输出，沿顶部绕行分发)
arrow(ax, 9.75, 1.50, 10.50, 1.50, style="-")
arrow(ax, 10.50, 1.50, 10.50, 7.60, style="-")
arrow(ax, 10.50, 7.60, 14.25, 7.60, style="-") # 水平总线贯穿 C 模块顶部
arrow(ax, 11.95, 7.60, 11.95, 7.10) # 垂直降落到 C1
arrow(ax, 14.25, 7.60, 14.25, 7.10) # 垂直降落到 C2

label(ax, 10.35, 4.55, "Best Model", size=8.5, bold=True, rotation=90)

# C1: Internal Validation (清除了奇怪的文字，完美对称)
rbox(ax, 11.95, 5.80, 2.10, 2.60)
label(ax, 11.95, 6.70, "Internal Validation", size=8.5, bold=True)
label(ax, 11.95, 6.35, "5-fold Stratified CV", size=7.5)
label(ax, 11.95, 6.10, "UCI dataset (n=303)", size=7.5)
rbox(ax, 11.95, 5.40, 1.80, 0.40, facecolor="#F5F5F5", lw=1.0, radius=0.04)
label(ax, 11.95, 5.40, "AUC = 0.762 ± 0.070", size=7.5, bold=True)

# C2: External Validation
rbox(ax, 14.25, 5.80, 2.10, 2.60)
label(ax, 14.25, 6.70, "External Validation", size=8.5, bold=True)
label(ax, 14.25, 6.35, "Kaggle Heart Failure", size=7.5)
label(ax, 14.25, 6.10, "n=918 · no retraining", size=7.5)
label(ax, 14.25, 5.85, "ca, thal → zero-padded", size=7.5, italic=True)
rbox(ax, 14.25, 5.40, 1.80, 0.40, facecolor="#E0E0E0", lw=1.2, radius=0.04)
label(ax, 14.25, 5.40, "AUC = 0.624 ± 0.035", size=7.5, bold=True)

# 2. 底部汇总总线 (C1与C2从底部流出，在中心汇入 Delta AUC)
arrow(ax, 11.95, 4.50, 11.95, 4.20, style="-")
arrow(ax, 14.25, 4.50, 14.25, 4.20, style="-")
arrow(ax, 11.95, 4.20, 14.25, 4.20, style="-")
arrow(ax, 13.10, 4.20, 13.10, 4.00)

# Delta AUC 框体
rbox(ax, 13.10, 3.80, 2.0, 0.40, facecolor=CLR_NODE_EMP, lw=1.2, radius=0.04)
label(ax, 13.10, 3.80, r"$\Delta AUC = −0.138$", size=8.5, bold=True)

# 3. 最后一击直达 Interpretability (独立在下方，无任何线穿透)
arrow(ax, 13.10, 3.60, 13.10, 3.10)

# Interpretability 框体
rbox(ax, 13.10, 2.30, 3.80, 1.60)
label(ax, 13.10, 2.85, "Interpretability Analysis", size=8.5, bold=True)
label(ax, 12.30, 2.45, "• Multi-head averaged attention", size=7.5, ha="left")
label(ax, 12.30, 2.15, "• Continuous variables prioritised", size=7.5, ha="left")

# 优化的灰度 Heatmap 视觉元素
cmap_vals = np.array([[0.9, 0.5, 0.2],[0.4, 0.8, 0.3],[0.1, 0.6, 0.95]])
for r in range(3):
    for c in range(3):
        intensity = cmap_vals[r, c]
        color_heat = plt.cm.Greys(intensity)
        rbox(ax, 11.60 + c*0.22, 2.08 + (2-r)*0.22, 0.20, 0.20,
             facecolor=color_heat, edgecolor=CLR_LINE, lw=0.5, radius=0.02)

# --- Final Export -------------------------------------------------------------
plt.tight_layout(pad=0.3)
plt.savefig("figure1_architecture.pdf", format='pdf', bbox_inches="tight", facecolor=CLR_BACK)
plt.savefig("figure1_architecture.png", format='png', dpi=300, bbox_inches="tight", facecolor=CLR_BACK)
plt.close()
print("Figure 1 (SCI Standard) has been successfully generated and saved as PDF & PNG.")
