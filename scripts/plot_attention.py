import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from train import GPT, build_model_config, DEPTH, device, apply_rotary_emb
from prepare import Tokenizer

# 强制使用纯英文字体，满足 SCI 要求
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 定义特征中英映射字典
# ==========================================
FEATURE_MAPPING = {
    "年龄": "Age",
    "性别": "Sex",
    "胸痛类型": "Chest Pain Type",
    "静息血压": "Resting BP",
    "胆固醇": "Cholesterol",
    "空腹血糖": "Fasting BS",
    "静息心电图": "Resting ECG",
    "最大心率": "Max HR",
    "运动诱发心绞痛": "Exercise Angina",
    "ST段压低": "Oldpeak",
    "ST段斜率": "ST Slope",
    "透视血管数": "Major Vessels",
    "地中海贫血": "Thalassemia",
    "最终诊断结果为": "Diagnosis"
}

def plot_attention_heatmap():
    # 1. 加载 Tokenizer
    tokenizer = Tokenizer.from_directory()

    # 2. 选取一个患病病人的测试文本
    sample_text = "患者特征：年龄63，性别1，胸痛类型4，静息血压145，胆固醇233，空腹血糖1，静息心电图2，最大心率150，运动诱发心绞痛0，ST段压低2.3，ST段斜率0，透视血管数0，地中海贫血1。最终诊断结果为："

    # 编码
    input_ids = tokenizer.enc.encode_ordinary(sample_text)
    input_ids.insert(0, tokenizer.get_bos_token_id())
    tokens = [tokenizer.enc.decode([idx]) for idx in input_ids]

    idx_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    B, T = idx_tensor.size()

    # 3. 加载模型
    config = build_model_config(DEPTH)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load("saved_models/model_fold1.pt", map_location=device))
    model.eval()

    # 4. 拦截最后一层注意力
    with torch.no_grad():
        cos_sin = model.cos[:, :T], model.sin[:, :T]
        x = model.transformer.wte(idx_tensor)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))

        x0 = x
        for i in range(len(model.transformer.h) - 1):
            block = model.transformer.h[i]
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](idx_tensor) if str(i) in model.value_embeds else None
            x = block(x, ve, cos_sin, model.window_sizes[i])

        i = len(model.transformer.h) - 1
        block = model.transformer.h[i]
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        x_norm = torch.nn.functional.rms_norm(x, (x.size(-1),))

        q = block.attn.c_q(x_norm).view(B, T, block.attn.n_head, block.attn.head_dim)
        k = block.attn.c_k(x_norm).view(B, T, block.attn.n_kv_head, block.attn.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 归一化修复
        q = torch.nn.functional.rms_norm(q, (q.size(-1),))
        k = torch.nn.functional.rms_norm(k, (k.size(-1),))

        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        attn_scores = (q_heads @ k_heads.transpose(-2, -1)) / (block.attn.head_dim ** 0.5)

        mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        attn_scores = attn_scores.masked_fill(~mask.view(1, 1, T, T), float('-inf'))
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()

    # 5. 提取注意力并映射到确定的英文特征槽位中
    final_step_attention = avg_attn_weights[-1, 1:]
    tokens_to_plot = tokens[1:]

    # 创建一个字典，用于累加每个“标准英文特征”的注意力
    feature_attention_sums = {en_name: 0.0 for en_name in FEATURE_MAPPING.values()}

    # 当前我们正在扫描哪个特征？（从“年龄”开始）
    current_feature_en = None

    for t, w in zip(tokens_to_plot, final_step_attention):
        clean_token = t.strip('：，。 ') # 去掉两端标点和空格

        # 1. 检查这个 token 是否包含“特征名”（例如“年龄”或者“年龄63”里的“年龄”）
        found_feature = False
        for zh_name, en_name in FEATURE_MAPPING.items():
            if zh_name in clean_token:
                current_feature_en = en_name
                found_feature = True
                break

        # 2. 如果我们处于一个特征的“管辖范围”内，就把权重加给它
        if current_feature_en:
            feature_attention_sums[current_feature_en] += w

    # 6. 构建 DataFrame 用于画图
    df_attn = pd.DataFrame({
        'Token_EN': list(feature_attention_sums.keys()),
        'Attention Weight': list(feature_attention_sums.values())
    })

    # 清理掉权重为 0 的特征（比如 Diagnosis）
    df_attn = df_attn[df_attn['Attention Weight'] > 0.0]
    df_attn = df_attn[df_attn['Token_EN'] != 'Diagnosis']
    # 按权重降序排列
    df_top = df_attn.sort_values(by='Attention Weight', ascending=False).head(12)

    # 7. 使用 Seaborn 绘制图表
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=df_top,
        x='Attention Weight',
        y='Token_EN',
        palette='Reds_r',
        edgecolor='black'
    )

    plt.title('Attention-based Feature Importance (Positive Case)', fontsize=16, fontweight='bold')
    plt.xlabel('Attention Weight (Probability)', fontsize=14, fontweight='bold')
    plt.ylabel('Clinical Features', fontsize=14, fontweight='bold')

    plt.grid(axis='x', linestyle=':', alpha=0.7, color='gray')
    plt.tight_layout()

    plt.savefig('attention_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig('attention_feature_importance.pdf', format='pdf', bbox_inches='tight')
    print("✓ 已生成全英文的临床特征重要性图表 (PNG 和 PDF)。")
    plt.show()

if __name__ == "__main__":
    plot_attention_heatmap()
