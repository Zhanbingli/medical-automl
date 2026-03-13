# import kagglehub

# print("开始从 Kaggle 下载 UCI 心脏病数据集...")
# # Download latest version
# path = kagglehub.dataset_download("mmoghadam10/uci-heart-disease-dataset")

# print("✅ 下载成功！数据保存在此路径:", path)


import kagglehub

# Download latest version
path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

print("Path to dataset files:", path)
