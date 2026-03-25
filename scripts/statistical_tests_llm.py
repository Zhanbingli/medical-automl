"""
scripts/statistical_tests_llm.py

扩展统计显著性检验，纳入 LLM 实验结果

检验方案：
  A. Custom Transformer vs LLM (各 shot 数) — UCI 5-fold 配对检验
  B. LLM 0-shot vs LLM N-shot — 验证 few-shot 效果是否显著
  C. Kaggle 泛化差距 — 描述性（无法配对，单次评估）

统计方法：
  - Wilcoxon 符号秩检验（非参数，适合小样本）
  - 配对 t 检验（与现有 statistical_tests.py 保持一致）
  - Cohen's d 效应量
  - Bonferroni 多重比较校正

Usage:
  uv run python scripts/statistical_tests_llm.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / "results"

# ── 数据加载 ───────────────────────────────────────────────────────────────────
def load_all():
    # Custom Transformer UCI 5-fold AUC
    tf = json.load(open(RESULTS / "results_kfold_5.json"))
    transformer_aucs = [r["val_auc"] for r in tf["individual_results"]]

    # LLM 0-shot 和 5-shot
    llm = json.load(open(RESULTS / "experiment_ollama_results.json"))
    llm_aucs = {
        0:  [r["auc"] for r in llm["uci_zero_shot"]["fold_results"]],
        5:  [r["auc"] for r in llm["uci_few_shot"]["fold_results"]],
    }
    llm_specs = {
        0:  [r["specificity"] for r in llm["uci_zero_shot"]["fold_results"]],
        5:  [r["specificity"] for r in llm["uci_few_shot"]["fold_results"]],
    }

    # Shot curve 1/3/10-shot
    sc = json.load(open(RESULTS / "experiment_shotcurve_results.json"))
    for n in [1, 3, 10]:
        folds = sorted(sc[str(n)]["uci_folds"], key=lambda x: x["fold"])
        llm_aucs[n]  = [r["auc"]         for r in folds]
        llm_specs[n] = [r["specificity"]  for r in folds]

    # Kaggle 单次评估（无法配对）
    kaggle = {
        "transformer": {
            "auc":  json.load(open(RESULTS / "external_validation_results.json"))["summary"]["auc"]["mean"],
            "spec": json.load(open(RESULTS / "external_validation_results.json"))["summary"]["specificity"]["mean"],
        },
        "llm": {}
    }
    kaggle["llm"][0]  = {"auc": llm["kaggle_zero_shot"]["auc"],
                         "spec": llm["kaggle_zero_shot"]["specificity"]}
    kaggle["llm"][5]  = {"auc": llm["kaggle_few_shot"]["auc"],
                         "spec": llm["kaggle_few_shot"]["specificity"]}
    for n in [1, 3, 10]:
        kaggle["llm"][n] = {"auc":  sc[str(n)]["kaggle"]["auc"],
                            "spec": sc[str(n)]["kaggle"]["specificity"]}

    return transformer_aucs, llm_aucs, llm_specs, kaggle


# ── 核心检验函数 ───────────────────────────────────────────────────────────────
def run_tests(a, b):
    """
    对两组配对数据运行 Wilcoxon + 配对 t-test，返回结果 dict。
    a: reference group (list of 5 floats)
    b: comparison group (list of 5 floats)
    mean_diff = mean(a) - mean(b)，正值表示 a 更好
    """
    a, b = np.array(a), np.array(b)
    diff = a - b

    # Wilcoxon 符号秩检验（n=5 时最小可达 p=0.0625，无法达到 p<0.05）
    try:
        w_stat, w_p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    except Exception:
        w_stat, w_p = np.nan, 1.0

    # 配对 t 检验
    t_stat, t_p = stats.ttest_rel(a, b)

    # Cohen's d（基于差值）
    std_diff = np.std(diff, ddof=1)
    cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else 0.0

    # 95% CI for mean difference
    n = len(diff)
    t_crit = stats.t.ppf(0.975, df=n-1)
    margin  = t_crit * std_diff / np.sqrt(n)
    ci = (float(np.mean(diff) - margin), float(np.mean(diff) + margin))

    def sig_label(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "n.s."

    def effect_label(d):
        d = abs(d)
        if d < 0.2: return "Negligible"
        if d < 0.5: return "Small"
        if d < 0.8: return "Medium"
        return "Large"

    return {
        "mean_a":    float(np.mean(a)),
        "mean_b":    float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "wilcoxon":  {"statistic": float(w_stat), "p": float(w_p), "sig": sig_label(w_p)},
        "ttest":     {"statistic": float(t_stat),  "p": float(t_p),  "sig": sig_label(t_p)},
        "cohens_d":  cohens_d,
        "effect":    effect_label(cohens_d),
        "ci_95":     ci,
        "n":         n,
    }


def bonferroni(results_dict, alpha=0.05):
    """在 results_dict 的 ttest.p 上做 Bonferroni 校正，添加 corrected_sig 字段。"""
    k = len(results_dict)
    for r in results_dict.values():
        p_corr = min(r["ttest"]["p"] * k, 1.0)
        r["ttest"]["p_bonferroni"] = p_corr
        r["ttest"]["sig_bonferroni"] = ("*" if p_corr < alpha else "n.s.")
    return results_dict


# ── 输出格式化 ────────────────────────────────────────────────────────────────
HDR = f"{'Comparison':<32} {'Ref Mean':>9} {'Cmp Mean':>9} {'Δ':>7} " \
      f"{'t-p':>8} {'t-sig':>6} {'W-p':>8} {'W-sig':>6} {'d':>7} {'Effect':<12} {'95% CI'}"
SEP = "─" * 120

def print_section(title, results_dict):
    print(f"\n{'═'*120}")
    print(f"  {title}")
    print(f"{'═'*120}")
    print(HDR)
    print(SEP)
    for name, r in results_dict.items():
        ci = r["ci_95"]
        print(
            f"  {name:<30} {r['mean_a']:>9.4f} {r['mean_b']:>9.4f} "
            f"{r['mean_diff']:>+7.4f} "
            f"{r['ttest']['p']:>8.4f} {r['ttest']['sig']:>6} "
            f"{r['wilcoxon']['p']:>8.4f} {r['wilcoxon']['sig']:>6} "
            f"{r['cohens_d']:>+7.3f} {r['effect']:<12} "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
    print(SEP)


def generate_latex(part_a, part_b):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical Significance Tests (Paired t-test, 5-Fold CV). "
        r"$^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$, n.s.~not significant. "
        r"Bonferroni-corrected p-values shown in parentheses.}",
        r"\label{tab:stat_llm}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Comparison & Ref AUC & Cmp AUC & $\Delta$ & $t$-test $p$ & Cohen's $d$ & 95\% CI \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Part A: Custom Transformer vs. LLM (UCI 5-fold)}} \\",
    ]
    for name, r in part_a.items():
        p  = r["ttest"]["p"]
        pb = r["ttest"].get("p_bonferroni", p)
        ps = r["ttest"]["sig"]
        pbs= r["ttest"].get("sig_bonferroni","")
        p_str = f"{p:.4f}{ps} ({pb:.4f}{pbs})" if pb != p else f"{p:.4f}{ps}"
        ci = r["ci_95"]
        lines.append(
            f"LLM {name} & {r['mean_a']:.3f} & {r['mean_b']:.3f} & "
            f"{r['mean_diff']:+.3f} & {p_str} & {r['cohens_d']:+.3f} & "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Part B: LLM 0-shot vs. LLM N-shot (Few-shot Calibration Effect)}} \\",
    ]
    for name, r in part_b.items():
        p  = r["ttest"]["p"]
        pb = r["ttest"].get("p_bonferroni", p)
        ps = r["ttest"]["sig"]
        pbs= r["ttest"].get("sig_bonferroni","")
        p_str = f"{p:.4f}{ps} ({pb:.4f}{pbs})" if pb != p else f"{p:.4f}{ps}"
        ci = r["ci_95"]
        lines.append(
            f"0-shot vs {name} & {r['mean_a']:.3f} & {r['mean_b']:.3f} & "
            f"{r['mean_diff']:+.3f} & {p_str} & {r['cohens_d']:+.3f} & "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 120)
    print("Statistical Significance Testing — LLM Extension")
    print("Wilcoxon Signed-Rank Test + Paired t-test + Cohen's d + Bonferroni Correction")
    print("=" * 120)

    transformer_aucs, llm_aucs, llm_specs, kaggle = load_all()

    print(f"\n  Custom Transformer UCI 5-fold AUC : "
          f"{np.mean(transformer_aucs):.4f} ± {np.std(transformer_aucs):.4f}")
    print(f"  Folds: {[round(x,4) for x in transformer_aucs]}")
    for n in [0, 1, 3, 5, 10]:
        print(f"  LLM {n}-shot UCI AUC: "
              f"{np.mean(llm_aucs[n]):.4f} ± {np.std(llm_aucs[n]):.4f}  "
              f"folds={[round(x,4) for x in llm_aucs[n]]}")

    # ── Part A: Transformer vs LLM ────────────────────────────────────────────
    part_a = {}
    for n in [0, 1, 3, 5, 10]:
        label = f"{n}-shot"
        part_a[label] = run_tests(transformer_aucs, llm_aucs[n])
    part_a = bonferroni(part_a)

    print_section("Part A: Custom Transformer vs. LLM (UCI 5-fold AUC, Ref = Transformer)", part_a)

    # ── Part B: LLM 0-shot vs N-shot (few-shot 校准效果) ──────────────────────
    part_b = {}
    for n in [1, 3, 5, 10]:
        label = f"{n}-shot"
        part_b[label] = run_tests(llm_aucs[0], llm_aucs[n])
    part_b = bonferroni(part_b)

    print_section("Part B: LLM 0-shot vs. N-shot (Few-shot Calibration Effect, Ref = 0-shot)", part_b)

    # ── Part C: Specificity 检验 ──────────────────────────────────────────────
    part_c = {}
    for n in [0, 1, 3, 5, 10]:
        part_c[f"LLM {n}-shot"] = run_tests(
            [0.0] * 5,   # placeholder — Transformer Kaggle spec is single value
            llm_specs[n]
        )
    # 重做：Transformer UCI spec vs LLM UCI spec
    tf_specs = json.load(open(RESULTS / "results_kfold_5.json"))
    tf_spec_vals = [r["val_spec"] for r in tf_specs["individual_results"]]

    part_c2 = {}
    for n in [0, 1, 3, 5, 10]:
        part_c2[f"{n}-shot"] = run_tests(tf_spec_vals, llm_specs[n])
    part_c2 = bonferroni(part_c2)

    print_section("Part C: Custom Transformer vs. LLM Specificity (UCI 5-fold, Ref = Transformer)", part_c2)

    # ── Part D: Kaggle 描述性（无配对，不做推断检验）────────────────────────────
    print(f"\n{'═'*120}")
    print("  Part D: Kaggle External Validation — Descriptive (No Paired Test, Single Evaluation)")
    print(f"{'═'*120}")
    print(f"  {'Model':<20} {'Kaggle AUC':>12} {'Δ vs Transformer':>18} {'Kaggle Spec':>12} {'Δ Spec':>10}")
    print("  " + "─" * 76)
    tf_kag_auc  = kaggle["transformer"]["auc"]
    tf_kag_spec = kaggle["transformer"]["spec"]
    print(f"  {'Custom Transformer':<20} {tf_kag_auc:>12.4f} {'—':>18} {tf_kag_spec:>12.4f} {'—':>10}")
    for n in [0, 1, 3, 5, 10]:
        kag = kaggle["llm"][n]
        da  = kag["auc"]  - tf_kag_auc
        ds  = kag["spec"] - tf_kag_spec
        print(f"  {'LLM ' + str(n) + '-shot':<20} {kag['auc']:>12.4f} "
              f"{da:>+18.4f} {kag['spec']:>12.4f} {ds:>+10.4f}")
    print("  " + "─" * 76)
    print("  Note: Kaggle is a single-point evaluation. "
          "Δ reported as descriptive evidence only.")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    output = {
        "method": "Wilcoxon signed-rank test + Paired t-test + Bonferroni correction",
        "n_folds": 5,
        "note_wilcoxon": "With n=5, minimum achievable Wilcoxon p = 0.0625 (cannot reach p<0.05)",
        "part_A_transformer_vs_llm": part_a,
        "part_B_fewshot_calibration": part_b,
        "part_C_specificity": part_c2,
        "part_D_kaggle_descriptive": {
            "transformer": kaggle["transformer"],
            "llm": {str(n): kaggle["llm"][n] for n in [0,1,3,5,10]},
        },
    }
    out_path = RESULTS / "statistical_tests_llm.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ 结果已保存: {out_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────
    latex = generate_latex(part_a, part_b)
    tex_path = RESULTS / "statistical_tests_llm.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"✓ LaTeX 表格已保存: {tex_path}")

    # ── 关键结论摘要 ──────────────────────────────────────────────────────────
    print(f"\n{'═'*120}")
    print("  KEY FINDINGS SUMMARY")
    print(f"{'═'*120}")

    r_5s = part_a["5-shot"]
    print(f"\n  1. Transformer vs LLM 5-shot (UCI AUC):")
    print(f"     Δ = {r_5s['mean_diff']:+.4f}  t-test p={r_5s['ttest']['p']:.4f} "
          f"({r_5s['ttest']['sig']})  d={r_5s['cohens_d']:+.3f} ({r_5s['effect']})")
    print(f"     → {'Transformer significantly better' if r_5s['mean_diff'] > 0 and r_5s['ttest']['p'] < 0.05 else 'No significant difference on UCI'}")

    r_0v5 = part_b["5-shot"]
    print(f"\n  2. LLM 0-shot vs 5-shot (Few-shot Calibration Effect):")
    print(f"     Δ = {r_0v5['mean_diff']:+.4f}  t-test p={r_0v5['ttest']['p']:.4f} "
          f"({r_0v5['ttest']['sig']})  d={r_0v5['cohens_d']:+.3f} ({r_0v5['effect']})")
    print(f"     → {'Significant calibration effect' if r_0v5['ttest']['p'] < 0.05 else 'Calibration effect present but not significant at α=0.05'}")

    print(f"\n  3. Kaggle AUC gap (descriptive, no paired test):")
    print(f"     Transformer: {tf_kag_auc:.4f}  |  LLM 5-shot: {kaggle['llm'][5]['auc']:.4f}  "
          f"Δ = {kaggle['llm'][5]['auc'] - tf_kag_auc:+.4f}")
    print(f"     LLM 5-shot shows {abs(kaggle['llm'][5]['auc'] - tf_kag_auc):.4f} AUC advantage on external validation")

    print(f"\n  ⚠  Wilcoxon note: With n=5 folds, minimum achievable p = 0.0625.")
    print(f"     Paired t-test used as primary test; Wilcoxon reported for robustness.")
    print(f"     Consider reporting both with this limitation noted in the paper.\n")


if __name__ == "__main__":
    main()
