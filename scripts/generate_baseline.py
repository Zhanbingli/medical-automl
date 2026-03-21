#!/usr/bin/env python3
"""
生成论文 Table X：两个队列的基线特征对比表
数据路径与 plot_dataset_shift.py 保持一致
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. 加载 UCI 数据
# ─────────────────────────────────────────
uci = pd.read_csv('patients.csv')
uci = uci.replace('?', np.nan)
for col in uci.columns:
    uci[col] = pd.to_numeric(uci[col], errors='coerce')

# 二值化 outcome
uci['label'] = (uci['num'] > 0).astype(int)

# ─────────────────────────────────────────
# 2. 加载 Kaggle 数据
# ─────────────────────────────────────────
kaggle_path = '/Users/lizhanbing12/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv'
kaggle = pd.read_csv(kaggle_path)

# 统一列名
kaggle = kaggle.rename(columns={
    'Age': 'age',
    'Sex': 'sex_str',
    'RestingBP': 'trestbps',
    'Cholesterol': 'chol',
    'FastingBS': 'fbs',
    'MaxHR': 'thalach',
    'ExerciseAngina': 'exang_str',
    'Oldpeak': 'oldpeak',
    'HeartDisease': 'label'
})
kaggle['sex']   = (kaggle['sex_str'] == 'M').astype(int)
kaggle['exang'] = (kaggle['exang_str'] == 'Y').astype(int)

# ─────────────────────────────────────────
# 3. 计算统计数据
# ─────────────────────────────────────────
def stat(series, is_binary=False):
    s = series.dropna()
    if is_binary:
        n = int(s.sum())
        pct = s.mean() * 100
        return f"{n} ({pct:.1f}%)"
    else:
        return f"{s.mean():.1f} ± {s.std():.1f}"

u, k = uci, kaggle

print("=" * 65)
print(f"{'Characteristic':<40} {'UCI (n=303)':<15} {'Kaggle (n=918)'}")
print("=" * 65)
print(f"{'[Demographics]'}")
print(f"{'Age, years (Mean ± SD)':<40} {stat(u['age']):<15} {stat(k['age'])}")
print(f"{'Male Sex, n (%)':<40} {stat(u['sex'], True):<15} {stat(k['sex'], True)}")
print()
print(f"{'[Clinical Features]'}")
print(f"{'Resting BP, mmHg (Mean ± SD)':<40} {stat(u['trestbps']):<15} {stat(k['trestbps'])}")
print(f"{'Cholesterol, mg/dL (Mean ± SD)':<40} {stat(u['chol']):<15} {stat(k['chol'])}")
print(f"{'Fasting Blood Sugar >120, n (%)':<40} {stat(u['fbs'], True):<15} {stat(k['fbs'], True)}")
print(f"{'Max Heart Rate, bpm (Mean ± SD)':<40} {stat(u['thalach']):<15} {stat(k['thalach'])}")
print(f"{'Exercise-Induced Angina, n (%)':<40} {stat(u['exang'], True):<15} {stat(k['exang'], True)}")
print(f"{'ST Depression Oldpeak (Mean ± SD)':<40} {stat(u['oldpeak']):<15} {stat(k['oldpeak'])}")
print()
print(f"{'[Features Absent in External Cohort]'}")
print(f"{'Major Vessels (ca)':<40} {'Available':<15} {'Zero-padded'}")
print(f"{'Thalassemia (thal)':<40} {'Available':<15} {'Zero-padded'}")
print()
print(f"{'[Missing Data]'}")
missing_uci = uci[['age','sex','trestbps','chol','fbs','thalach','exang','oldpeak']].isnull().any(axis=1).sum()
print(f"{'Records with missing values, n (%)':<40} {missing_uci} ({missing_uci/303*100:.1f}%)  {'0 (0.0%)'}")
print()
print(f"{'[Outcome]'}")
pos_u = int(u['label'].sum());  n_u = len(u)
pos_k = int(k['label'].sum());  n_k = len(k)
print(f"{'CVD Positive, n (%)':<40} {pos_u} ({pos_u/n_u*100:.1f}%)   {pos_k} ({pos_k/n_k*100:.1f}%)")
print(f"{'CVD Negative, n (%)':<40} {n_u-pos_u} ({(n_u-pos_u)/n_u*100:.1f}%)   {n_k-pos_k} ({(n_k-pos_k)/n_k*100:.1f}%)")
print("=" * 65)
print("\n✅ 请将以上数值复制到论文 Section 2.1 末尾的 Table 中")
