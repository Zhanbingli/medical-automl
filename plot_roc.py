import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import os

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def generate_roc_comparison():
    fig, ax = plt.subplots(figsize=(8, 6))

    # ==========================================
    # 1. 重现传统基线 (Baseline) 的数据
    # ==========================================
    df = pd.read_csv('patients.csv').replace('?', np.nan).dropna()
    X = df.drop(['Index', 'num'], axis=1).astype(float)
    y = df['num'].apply(lambda x: 1 if int(x) > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 画 Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    ax.plot(fpr_rf, tpr_rf, color='#2ca02c', linestyle='--', linewidth=2, label=f'Random Forest (AUC = {auc_rf:.3f})')

    # 画 XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
    auc_xgb = auc(fpr_xgb, tpr_xgb)
    ax.plot(fpr_xgb, tpr_xgb, color='#1f77b4', linestyle='-.', linewidth=2, label=f'XGBoost (AUC = {auc_xgb:.3f})')

    # ==========================================
    # 2. 读取我们大模型 (Transformer) 的数据
    # ==========================================
    npz_path = os.path.join("data", "latest_predictions.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        y_true_llm = data['y_true']
        y_prob_llm = data['y_prob']

        fpr_llm, tpr_llm, _ = roc_curve(y_true_llm, y_prob_llm)
        auc_llm = auc(fpr_llm, tpr_llm)

        # 大模型用最显眼的红色实线
        ax.plot(fpr_llm, tpr_llm, color='#d62728', linewidth=2.5, label=f'Agent-Optimized LLM (AUC = {auc_llm:.3f})')
    else:
        print("⚠️ 未找到大模型的预测数据，请先运行一次 train.py。")

    # ==========================================
    # 3. 装饰图表
    # ==========================================
    # 画一条对角虚线 (代表瞎猜的基线 0.5)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5)

    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc="lower right", frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig('roc_comparison_curve.png', dpi=300, bbox_inches='tight')
    print("✅ ROC 对比曲线已生成：roc_comparison_curve.png")

if __name__ == "__main__":
    generate_roc_comparison()
