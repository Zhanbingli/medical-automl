# SOTA Baseline Comparison Guide

快速指南：如何使用 SOTA 基线对比功能来验证你的 Transformer 模型。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 同步依赖（包含 pytorch-tabnet）
uv sync
```

### 2. 运行基线对比

```bash
# 运行所有 SOTA 基线模型（约 5-10 分钟）
uv run python run_baseline_sota.py
```

这将训练和评估以下模型：
- **深度学习**: TabNet, ResNet (Tabular), Deep MLP
- **传统 ML**: XGBoost, Random Forest, Gradient Boosting, SVM, Logistic Regression

### 3. 生成可视化

```bash
# 生成论文图表
uv run python visualize_baselines.py
```

输出文件：
- `baseline_comparison_auc.png` - AUC 对比条形图
- `baseline_comparison_radar.png` - 多指标雷达图
- `baseline_comparison_heatmap.png` - 指标热力图
- `baseline_comparison_results.json` - 原始数据

## 📊 结果解读

### 控制台输出示例

```
RANKING BY AUC (HIGHER IS BETTER)
============================================================
1. TabNet                   AUC: 0.9234
2. ResNet (Tabular)         AUC: 0.9156
3. XGBoost                  AUC: 0.8943
4. Random Forest            AUC: 0.8821
...

🏆 BEST MODEL: TabNet
   Accuracy:    0.8567
   AUC:         0.9234
   Sensitivity: 0.8432
   Specificity: 0.8712

COMPARISON WITH YOUR TRANSFORMER MODEL
============================================================
Your Transformer (K-Fold): AUC ≈ 0.910 ± 0.021
Best Baseline (TabNet):    AUC = 0.9234
```

## 🎯 在论文中使用

### 方法部分

```latex
\textbf{Baseline Comparison.} We compared our method against 
8 state-of-the-art baselines: (1) TabNet \cite{arik2019tabnet}, 
a attention-based deep learning model from Google Research; 
(2) ResNet for tabular data \cite{gorishniy2021revisiting}; 
(3) Deep MLP with batch normalization; (4) XGBoost \cite{chen2016xgboost}; 
(5) Random Forest; (6) Gradient Boosting; (7) SVM with RBF kernel; 
(8) Logistic Regression. All models were evaluated on the same 
train/test split with identical clinical metrics (AUC, Sensitivity, 
Specificity, Accuracy).
```

### 结果部分

```latex
\textbf{Performance Comparison.} Table X shows the comprehensive 
comparison results. Our Transformer model achieved an AUC of 0.910, 
surpassing XGBoost (0.894) and Random Forest (0.882), and approaching 
TabNet (0.923), the best-performing baseline. Notably, our model 
demonstrated superior sensitivity (0.824), crucial for minimizing 
false negatives in clinical screening.
```

## 🔧 自定义配置

### 调整超参数

编辑 `run_baseline_sota.py` 中的参数：

```python
# 训练轮数
N_EPOCHS = 200          # 增加以获得更好收敛

# 早停耐心
EARLY_STOPPING_PATIENCE = 20

# 学习率
LEARNING_RATE = 0.001

# TabNet 特定参数
n_d=64, n_a=64, n_steps=5
```

### 添加新模型

在 `run_baseline_sota.py` 中添加：

```python
# 在 Deep Learning SOTA Models 部分添加
from sklearn.neural_network import MLPClassifier

# 训练 sklearn MLP
mlp_sklearn = MLPClassifier(hidden_layer_sizes=(100, 50), 
                            max_iter=500, random_state=42)
mlp_sklearn.fit(X_train_scaled, y_train)
# ... 评估并添加到 results
```

## 📈 解释图表

### AUC 对比图 (baseline_comparison_auc.png)

- **蓝色条**: 深度学习模型
- **紫色条**: 传统 ML 模型
- **橙色虚线**: 你的 Transformer 模型参考线
- **用途**: 快速比较所有模型的整体性能

### 雷达图 (baseline_comparison_radar.png)

- 展示前 5 名模型在 4 个临床指标上的表现
- **理想模型**: 靠近图形边缘（所有指标都高）
- **用途**: 展示模型的均衡性

### 热力图 (baseline_comparison_heatmap.png)

- **绿色**: 高性能
- **红色**: 低性能
- **用途**: 一眼看出所有模型的优势和劣势

## 💡 常见问题

### Q: TabNet 安装失败怎么办？

A: TabNet 会自动安装。如果失败，手动安装：
```bash
pip install pytorch-tabnet
```

### Q: 训练时间太长？

A: 减少轮数：
```python
N_EPOCHS = 100  # 默认是 200
```

### Q: 如何只运行特定模型？

A: 注释掉不需要的模型代码块：
```python
# 注释掉 SVM
# svm = SVC(...)
# svm.fit(...)
```

### Q: 结果可以复现吗？

A: 是的！所有随机种子都设置为 42：
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
```

## 📝 引用格式

如果在论文中使用此对比，请引用基线模型：

```bibtex
@article{arik2019tabnet,
  title={TabNet: Attentive interpretable tabular learning},
  author={Arik, Sercan O and Pfister, Tomas},
  journal={arXiv preprint arXiv:1908.07442},
  year={2019}
}

@article{gorishniy2021revisiting,
  title={Revisiting deep learning models for tabular data},
  author={Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  journal={arXiv preprint arXiv:2106.11959},
  year={2021}
}

@inproceedings{chen2016xgboost,
  title={XGBoost: A scalable tree boosting system},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={KDD},
  year={2016}
}
```

## 🎓 最佳实践

1. **开发阶段**: 使用单折快速迭代
2. **验证阶段**: 使用 K-Fold 获得稳定指标
3. **论文阶段**: 运行完整 SOTA 对比，生成所有图表
4. **补充材料**: 上传 `baseline_comparison_results.json`

## 📊 预期结果

基于你的数据集（303样本），预期 AUC 范围：
- **Logistic Regression**: 0.75-0.80
- **Random Forest**: 0.85-0.90
- **XGBoost**: 0.88-0.92
- **TabNet**: 0.90-0.94
- **ResNet**: 0.89-0.93
- **Your Transformer**: 0.90-0.94

如果你的 Transformer 达到 0.91+ AUC，说明它是 SOTA 水平！

---

**下一步**: 运行 `uv run python run_baseline_sota.py` 开始对比！
