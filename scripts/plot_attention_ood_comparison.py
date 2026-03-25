"""
scripts/plot_attention_ood_comparison.py

Figure: Attention Weight Disruption Under OOD Zero-Padding

Two-panel comparison showing how the Transformer's last-layer attention
weights (at the diagnosis token position) change when ca and thal features
are zero-padded as in the Kaggle external cohort.

Left panel  — UCI-style input: ca=2 (genuine clinical finding), thal=7 (reversible defect)
Right panel — Kaggle-style input: same patient with ca=0, thal=0 (missing-value imputation)

A third panel shows the absolute attention shift (Right − Left), highlighting
which features are most disrupted by the OOD encoding.

Usage:
    uv run python scripts/plot_attention_ood_comparison.py
"""

import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train import GPT, build_model_config, DEPTH, apply_rotary_emb
from prepare import Tokenizer

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Feature label mapping (Chinese token → English display label) ─────────────
FEATURE_MAP = {
    "年龄":       "Age",
    "性别":       "Sex",
    "胸痛类型":   "Chest Pain Type",
    "静息血压":   "Resting BP",
    "胆固醇":     "Cholesterol",
    "空腹血糖":   "Fasting BS",
    "静息心电图": "Resting ECG",
    "最大心率":   "Max HR",
    "运动诱发心绞痛": "Exercise Angina",
    "ST段压低":   "Oldpeak",
    "ST段斜率":   "ST Slope",
    "透视血管数": "Major Vessels",  # ca
    "地中海贫血": "Thalassemia",    # thal
    "最终诊断结果为": "Diagnosis",
}
FEATURE_ORDER = [v for v in FEATURE_MAP.values() if v != "Diagnosis"]

# ── Patient inputs ─────────────────────────────────────────────────────────────
# UCI-style: ca=2 (2 fluoroscopically visible vessels), thal=7 (reversible defect)
# Typical high-risk positive patient
PATIENT_UCI = (
    "患者特征：年龄58，性别1，胸痛类型4，静息血压130，胆固醇236，"
    "空腹血糖0，静息心电图2，最大心率174，运动诱发心绞痛0，"
    "ST段压低0，ST段斜率2，透视血管数2，地中海贫血7。最终诊断结果为："
)

# Kaggle-style: same patient but ca=0, thal=0 (zero-padded missing values)
PATIENT_KAGGLE = (
    "患者特征：年龄58，性别1，胸痛类型4，静息血压130，胆固醇236，"
    "空腹血糖0，静息心电图2，最大心率174，运动诱发心绞痛0，"
    "ST段压低0，ST段斜率2，透视血管数0，地中海贫血0。最终诊断结果为："
)

LABELS = {
    "uci":    "UCI-style Input\n(ca=2, thal=7  — genuine clinical values)",
    "kaggle": "Kaggle-style Input\n(ca=0, thal=0  — zero-padded missing values)",
}


# ── Core: extract per-feature attention at the last token position ─────────────
def extract_feature_attention(model, tokenizer, text: str) -> dict:
    """
    Forward through all layers except the last manually, then compute
    raw Q·K attention scores at the final (diagnosis) token position.
    Returns a dict mapping English feature label → aggregated attention weight.
    """
    input_ids = tokenizer.enc.encode_ordinary(text)
    input_ids.insert(0, tokenizer.get_bos_token_id())
    tokens_decoded = [tokenizer.enc.decode([i]) for i in input_ids]
    T = len(input_ids)
    idx_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    B = 1

    with torch.no_grad():
        cos_sin = model.cos[:, :T], model.sin[:, :T]

        # Embedding + layer-norm
        x = model.transformer.wte(idx_tensor)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        # Run all layers except last
        n_layers = len(model.transformer.h)
        for i in range(n_layers - 1):
            block = model.transformer.h[i]
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = (model.value_embeds[str(i)](idx_tensor)
                  if str(i) in model.value_embeds else None)
            x = block(x, ve, cos_sin, model.window_sizes[i])

        # Last layer: extract Q and K manually
        i = n_layers - 1
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        x_norm = F.rms_norm(x, (x.size(-1),))

        block = model.transformer.h[i]
        q = block.attn.c_q(x_norm).view(B, T, block.attn.n_head, block.attn.head_dim)
        k = block.attn.c_k(x_norm).view(B, T, block.attn.n_kv_head, block.attn.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Expand KV heads to match Q heads (GQA)
        k = k.repeat_interleave(block.attn.n_head // block.attn.n_kv_head, dim=2)

        q_h = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k_h = k.transpose(1, 2)

        # Attention scores: (B, n_head, T, T)
        scores = (q_h @ k_h.transpose(-2, -1)) / (block.attn.head_dim ** 0.5)

        # Causal mask
        mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax → weights
        weights = F.softmax(scores, dim=-1)

        # Last token (diagnosis position), averaged across heads: shape (T,)
        last_token_attn = weights[0, :, -1, :].mean(dim=0).cpu().numpy()  # (T,)

    # ── Aggregate attention per feature ──────────────────────────────────────
    feature_attn = {en: 0.0 for en in FEATURE_ORDER}
    current_feature = None

    for token_str, w in zip(tokens_decoded, last_token_attn):
        clean = token_str.strip('：，。 ')
        for zh, en in FEATURE_MAP.items():
            if zh in clean and en != "Diagnosis":
                current_feature = en
                break
        if current_feature and current_feature != "Diagnosis":
            feature_attn[current_feature] += float(w)

    # Normalise so bars sum to 1 (excluding diagnosis token itself)
    total = sum(feature_attn.values())
    if total > 0:
        feature_attn = {k: v / total for k, v in feature_attn.items()}

    return feature_attn


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot(uci_attn: dict, kag_attn: dict):
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})

    features = FEATURE_ORDER
    x = np.arange(len(features))
    uci_vals = np.array([uci_attn[f] for f in features])
    kag_vals = np.array([kag_attn[f] for f in features])
    delta    = kag_vals - uci_vals

    # Colour by OOD sensitivity
    bar_colors_uci = ["#d73027" if f in ("Major Vessels", "Thalassemia")
                      else "#4575b4" for f in features]
    bar_colors_kag = ["#d73027" if f in ("Major Vessels", "Thalassemia")
                      else "#4575b4" for f in features]
    delta_colors   = ["#b2182b" if d > 0 else "#2166ac" for d in delta]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Attention Weight Disruption Under OOD Zero-Padding\n"
        "(Last-layer attention at diagnosis token position, averaged across heads)",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.55, figure=fig)

    # ── Panel A: UCI ──────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(x, uci_vals, color=bar_colors_uci, alpha=0.85,
                   edgecolor="white", linewidth=0.8)
    ax1.set_title(f"(A) {LABELS['uci']}", fontsize=11, fontweight="bold", loc="left")
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("Normalised\nAttention Weight", fontsize=10)
    ax1.set_ylim(0, max(uci_vals.max(), kag_vals.max()) * 1.25)
    # Annotate OOD features
    for feat, bar in zip(features, bars):
        if feat in ("Major Vessels", "Thalassemia"):
            ax1.annotate("[OOD]", xy=(bar.get_x() + bar.get_width() / 2,
                                      bar.get_height() + 0.003),
                         ha="center", fontsize=7.5, color="#b2182b",
                         fontweight="bold")

    # ── Panel B: Kaggle ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.bar(x, kag_vals, color=bar_colors_kag, alpha=0.85,
                    edgecolor="white", linewidth=0.8)
    ax2.set_title(f"(B) {LABELS['kaggle']}", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xticks(x)
    ax2.set_xticklabels(features, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Normalised\nAttention Weight", fontsize=10)
    ax2.set_ylim(0, max(uci_vals.max(), kag_vals.max()) * 1.25)
    for feat, bar in zip(features, bars2):
        if feat in ("Major Vessels", "Thalassemia"):
            ax2.annotate("[OOD]", xy=(bar.get_x() + bar.get_width() / 2,
                                      bar.get_height() + 0.003),
                         ha="center", fontsize=7.5, color="#b2182b",
                         fontweight="bold")

    # ── Panel C: Delta ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(x, delta, color=delta_colors, alpha=0.85,
            edgecolor="white", linewidth=0.8)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_title(
        "(C) Attention Shift: Kaggle − UCI  "
        "(red ↑ = OOD token attracts more attention; blue ↓ = other features lose attention)",
        fontsize=11, fontweight="bold", loc="left"
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(features, rotation=35, ha="right", fontsize=9)
    ax3.set_ylabel("Δ Attention\n(Kaggle − UCI)", fontsize=10)

    # Annotate largest shifts
    for i_f, (feat, d) in enumerate(zip(features, delta)):
        if abs(d) > 0.02:
            ax3.annotate(
                f"{d:+.3f}",
                xy=(i_f, d + (0.004 if d > 0 else -0.008)),
                ha="center", fontsize=8,
                color="#b2182b" if d > 0 else "#2166ac",
                fontweight="bold"
            )

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", alpha=0.85, label="OOD-impacted features (ca / thal)"),
        Patch(facecolor="#4575b4", alpha=0.85, label="Other clinical features"),
        Patch(facecolor="#b2182b", alpha=0.85, label="Δ > 0: attention increase under OOD"),
        Patch(facecolor="#2166ac", alpha=0.85, label="Δ < 0: attention loss (routing disruption)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    fig.subplots_adjust(top=0.93, bottom=0.12, hspace=0.65)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES, f"attention_ood_comparison.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_directory()

    print("Loading model (fold1)...")
    config = build_model_config(DEPTH)
    model = GPT(config).to(device)
    ckpt = os.path.join(ROOT, "saved_models", "model_fold1.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"  Model loaded from {ckpt}")

    print("\nExtracting attention — UCI-style input...")
    uci_attn = extract_feature_attention(model, tokenizer, PATIENT_UCI)
    print("  Feature attention (UCI):")
    for k, v in uci_attn.items():
        print(f"    {k:22s}: {v:.4f}")

    print("\nExtracting attention — Kaggle-style input (ca=0, thal=0)...")
    kag_attn = extract_feature_attention(model, tokenizer, PATIENT_KAGGLE)
    print("  Feature attention (Kaggle):")
    for k, v in kag_attn.items():
        print(f"    {k:22s}: {v:.4f}")

    print("\nPlotting comparison figure...")
    plot(uci_attn, kag_attn)

    print("\nDelta (Kaggle − UCI):")
    for f in FEATURE_ORDER:
        d = kag_attn[f] - uci_attn[f]
        bar = "▲" if d > 0 else "▼"
        print(f"  {f:22s}: {d:+.4f}  {bar}")

    print("\nDone. View: open figures/attention_ood_comparison.png")
