# Statistical Testing Quick Reference

快速指南：如何运行统计显著性检验来验证你的 Transformer 是否显著优于基线。

## 🚀 快速开始

### 前提条件

确保你已经运行了：
1. `uv run python run_baseline_sota.py` （生成 baseline_comparison_5fold.json）
2. `uv run python train_kfold.py` （生成 results_kfold_5.json，可选）

### 运行统计检验

```bash
uv run python statistical_tests.py
```

### 交互式输入（如果没有 JSON 文件）

如果没有找到 Transformer 的 fold 结果，脚本会提示你手动输入：

```
Warning: Transformer fold results not found in JSON files.
Please manually enter the 5 fold AUC values:
  Fold 1 AUC: 0.941
  Fold 2 AUC: 0.926
  Fold 3 AUC: 0.918
  Fold 4 AUC: 0.905
  Fold 5 AUC: 0.912
```

## 📊 输出解读

### 控制台输出示例

```
====================================================================================================
WILCOXON SIGNED-RANK TEST RESULTS
Transformer vs. Baseline Models (5-Fold CV)
====================================================================================================

Comparison                     Mean Diff   p-value      Sig    Effect r   CI 95%                    Interpretation
----------------------------------------------------------------------------------------------------
vs TabNet                      +0.0012     0.2345       n.s.   +0.1346    [-0.0213, +0.0241]        Small
vs ResNet (Tabular)            +0.0089     0.1234       n.s.   +0.2341    [-0.0156, +0.0324]        Small
vs Random Forest               +0.0345     0.0089       **     +0.5678    [+0.0123, +0.0567]        Large
vs XGBoost                     +0.0289     0.0156       *      +0.4890    [+0.0089, +0.0489]        Medium
...
----------------------------------------------------------------------------------------------------

Significance: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant
Effect Size:  |r|<0.1 Negligible, <0.3 Small, <0.5 Medium, ≥0.5 Large
CI 95%:       Bootstrap confidence interval for mean difference
====================================================================================================
```

### 关键指标解释

| 指标 | 含义 | 如何判断 |
|------|------|---------|
| **Mean Diff** | Transformer - Baseline 的平均 AUC 差 | + 表示 Transformer 更好，- 表示更差 |
| **p-value** | 显著性水平 | <0.05 为显著，<0.01 为非常显著 |
| **Sig** | 显著性标记 | ***/**/* = 显著，n.s. = 不显著 |
| **Effect r** | 效应量 (rank-biserial correlation) | 0.1/0.3/0.5 是小/中/大效应 |
| **CI 95%** | 95% 置信区间 | 不包含 0 说明差异显著 |

## 📝 在论文中报告

### 方法部分

```latex
\textbf{Statistical Analysis.} We performed Wilcoxon signed-rank tests 
to compare the Transformer model against baseline models using paired 
5-fold cross-validation results. Effect sizes were calculated as 
rank-biserial correlations ($r$), and 95\% confidence intervals were 
estimated using bootstrap resampling (10,000 iterations). Statistical 
significance was set at $\alpha = 0.05$.
```

### 结果部分

```latex
\textbf{Statistical Significance.} Table \ref{tab:statistical_tests} 
summarizes the statistical comparison results. The Transformer achieved 
significantly higher AUC than XGBoost ($p = 0.009$, $r = 0.57$, 
medium-large effect) and Random Forest ($p = 0.016$, $r = 0.49$, 
medium effect), while showing no significant difference compared to 
TabNet ($p = 0.23$, $r = 0.13$, small effect).
```

### LaTeX 表格

运行脚本后会自动生成 `statistical_tests_table.tex`：

```latex
\begin{table}[htbp]
\centering
\caption{Wilcoxon Signed-Rank Test: Transformer vs. Baseline Models}
\label{tab:statistical_tests}
\begin{tabular}{lcccc}
\toprule
Model & Mean Diff & p-value & Effect $r$ & 95\% CI \\
\midrule
TabNet & +0.001 & 0.2345 & +0.135 & [-0.021, +0.024] \\
ResNet & +0.009 & 0.1234 & +0.234 & [-0.016, +0.032] \\
Random Forest & +0.035 & 0.0089** & +0.568 & [+0.012, +0.057] \\
XGBoost & +0.029 & 0.0156* & +0.489 & [+0.009, +0.049] \\
\bottomrule
\end{tabular}
\end{table}
```

## 🎯 如何判断结果

### ✅ 理想情况

```
vs Best Baseline (TabNet):  p = 0.23 (n.s.), r = 0.13
```
- **解读**: Transformer 与最佳基线性能相当（无显著差异）
- **效应量**: 小，实际意义不大
- **论文写法**: "comparable to the state-of-the-art TabNet baseline"

### 🎉 显著优于基线

```
vs XGBoost:  p = 0.009 (**), r = 0.57, CI = [+0.012, +0.057]
```
- **解读**: Transformer 显著优于 XGBoost（p < 0.01）
- **效应量**: 中-大，实际意义重大
- **论文写法**: "significantly outperformed XGBoost (p = 0.009, medium-large effect)"

### ⚠️ 需要改进

```
vs TabNet:  p = 0.03 (*), r = -0.45, CI = [-0.051, -0.003]
```
- **解读**: Transformer 显著差于 TabNet
- **效应量**: 中等
- **建议**: 继续优化架构或训练策略

## 📁 输出文件

运行后会生成：

1. **statistical_tests_results.json** - 详细结果
   ```json
   {
     "test_method": "Wilcoxon Signed-Rank Test",
     "comparisons": {
       "TabNet": {
         "p_value": 0.2345,
         "effect_size_r": 0.1346,
         "confidence_interval_95": {"lower": -0.0213, "upper": 0.0241},
         ...
       }
     }
   }
   ```

2. **statistical_tests_table.tex** - LaTeX 表格代码

## 💡 常见问题

### Q: p-value 不显著怎么办？

**A**: 不显著不代表你的模型不好！可能说明：
- 你的模型与 SOTA 基线性能相当（这是好事！）
- 样本量太小（5 folds），统计功效不足
- 效应量小，实际意义不大

**论文写法**: "achieved comparable performance to TabNet without significant difference"

### Q: 效应量和 p-value 矛盾？

**A**: 可能的情况：
- **大效应量 + 不显著 p**: 样本量太小，需要更多 folds
- **小效应量 + 显著 p**: 差异存在但实际意义不大

**优先看效应量**，它反映了实际重要性。

### Q: 为什么用 Wilcoxon 而不是 t-test？

**A**: 
- Wilcoxon 是非参数检验，不要求正态分布
- 适合小样本（5 folds）
- 对异常值不敏感
- 医学统计中更常用

### Q: 可以做多重比较校正吗？

**A**: 可以！如果比较很多基线，建议用 Bonferroni 校正：
- 比较 5 个基线，显著性水平改为 0.05/5 = 0.01
- 或者使用 FDR (False Discovery Rate)

当前脚本未自动校正，你可以在解读时手动调整。

## 🔬 技术细节

### Wilcoxon Signed-Rank Test

**适用条件**：
- 配对样本（同一 folds 比较）✅
- 连续变量（AUC）✅
- 样本量小（n=5）✅

**原假设 H₀**: Transformer 和 Baseline 的 AUC 中位数相等  
**备择假设 H₁**: 两者中位数不等

### Effect Size (Rank-Biserial Correlation)

**计算公式**:  
r = Z / √(2N)

其中 Z 是标准化统计量，N 是样本数

**解释**（Cohen 标准）：
- |r| < 0.1: 可忽略
- 0.1 ≤ |r| < 0.3: 小效应
- 0.3 ≤ |r| < 0.5: 中效应
- |r| ≥ 0.5: 大效应

### Bootstrap Confidence Interval

**方法**: Percentile Bootstrap  
**迭代次数**: 10,000  
**种子**: 42 (可复现)

**计算步骤**:
1. 从 5 个差异值中有放回地随机抽样
2. 计算每次抽样的平均差异
3. 重复 10,000 次
4. 取 2.5% 和 97.5% 分位数作为 95% CI

## 📚 推荐阅读

1. **Wilcoxon Test**: Wilcoxon, F. (1945). Individual comparisons by ranking methods
2. **Effect Size**: Kerby, D. S. (2014). The simple difference formula: An approach to teaching nonparametric correlation
3. **Bootstrap CI**: Efron, B. (1987). Better bootstrap confidence intervals

## 🚀 下一步

运行以下命令完成完整的统计验证：

```bash
# 1. 运行基线对比（如果没有）
uv run python run_baseline_sota.py

# 2. 运行统计检验
uv run python statistical_tests.py

# 3. 查看 LaTeX 表格
cat statistical_tests_table.tex

# 4. 使用 LaTeX 表格到你的论文
```

---

**现在运行**: `uv run python statistical_tests.py`
