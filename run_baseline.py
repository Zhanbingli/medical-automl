import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

print("正在加载并清洗患者数据...")
# 1. 读取数据
df = pd.read_csv('patients.csv')

# --- 新增的清洗步骤 ---
# 将文本 '?' 替换为标准的缺失值 NaN，并直接丢弃包含缺失值的病人记录
df = df.replace('?', np.nan).dropna()

# 2. 划分特征与标签
# 强制把所有特征列转换为浮点数矩阵，以满足传统算法的纯数值要求
X = df.drop(['Index', 'num'], axis=1).astype(float)
y = df['num'].apply(lambda x: 1 if int(x) > 0 else 0) # 标签二值化

# 3. 划分数据集 (9:1比例)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 4. 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# 5. 训练 XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# 6. 打印临床级评估报告
def print_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred) # 敏感度 (Recall)
    auc = roc_auc_score(y_true, y_prob)
    print(f"--- {name} ---")
    print(f"准确率 (Accuracy):   {acc:.4f}")
    print(f"敏感度 (Sensitivity): {sens:.4f}")
    print(f"AUC 面积:            {auc:.4f}\n")

print("传统机器学习基线运行结果：")
print_metrics("Random Forest", y_test, rf_preds, rf_probs)
print_metrics("XGBoost", y_test, xgb_preds, xgb_probs)
