import pandas as pd
df = pd.read_csv('/Users/lizhanbing12/.cache/kagglehub/datasets/fedesoriano/heart-failure-prediction/versions/1/heart.csv')
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df['HeartDisease'].value_counts())
