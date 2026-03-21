import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 SCI 字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

def clean_data(df, cols):
    """清洗数据：处理问号并转换为浮点数，保持与训练脚本一致"""
    df = df.copy()
    df = df.replace('?', np.nan)
    for col in cols:
        if col in df.columns:
            # 填充缺失值为众数
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            # 强制转换为数值类型以满足 kdeplot 要求
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def plot_distribution_shift():
    # 1. 加载并清洗 UCI 数据
    try:
        uci_df = pd.read_csv('patients.csv')
        features_to_plot = ['age', 'chol', 'thalach', 'ca']
        uci_df = clean_data(uci_df, features_to_plot)
    except Exception as e:
        print(f"Error loading UCI data: {e}")
        return

    # 2. 加载并清洗外部 Kaggle 数据
    try:
        external_path = '/Users/lizhanbing12/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv'
        external_df = pd.read_csv(external_path)

        # 重命名列名以对齐 UCI 格式
        external_df = external_df.rename(columns={
            'Age': 'age',
            'Cholesterol': 'chol',
            'MaxHR': 'thalach'
        })

        # 补零逻辑：Kaggle 缺 ca 和 thal，强行补 0
        if 'ca' not in external_df.columns:
            external_df['ca'] = 0

        # 同样进行类型转换
        external_df = clean_data(external_df, features_to_plot)
    except Exception as e:
        print(f"Error loading External data: {e}")
        return

    # 3. 绘图
    feature_names = ['Age', 'Cholesterol', 'Max Heart Rate', 'Major Vessels (ca)']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    for i, col in enumerate(features_to_plot):
        # 绘制 UCI 分布 (训练集)
        sns.kdeplot(uci_df[col].dropna(), ax=axes[i], fill=True, color='#d62728',
                    label='UCI (Training)', alpha=0.5, linewidth=2)
        # 绘制 External 分布 (测试集)
        sns.kdeplot(external_df[col].dropna(), ax=axes[i], fill=True, color='#1f77b4',
                    label='Kaggle (External)', alpha=0.5, linewidth=2)

        axes[i].set_title(f'Distribution Shift: {feature_names[i]}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Value', fontsize=12)
        axes[i].set_ylabel('Density', fontsize=12)
        axes[i].legend()
        axes[i].grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig('dataset_distribution_shift.png', dpi=300, bbox_inches='tight')
    plt.savefig('dataset_distribution_shift.pdf', format='pdf', bbox_inches='tight')
    print("✓ 成功生成数据集偏移对比图：dataset_distribution_shift.png")

if __name__ == "__main__":
    plot_distribution_shift()
