import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train import GPT, build_model_config, DEPTH, device, apply_rotary_emb
from prepare import Tokenizer
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号
def plot_attention_heatmap():
    # 1. 加载 Tokenizer
    tokenizer = Tokenizer.from_directory()

    # 2. 选取一个患病病人的测试文本（这里你可以替换成任意一个阳性病人的真实文本）
   # 2. 选取一个患病病人的测试文本（严格对齐 prepare.py 中的中文模板！）
    sample_text = "患者特征：年龄63，性别1，胸痛类型4，静息血压145，胆固醇233，空腹血糖1，静息心电图2，最大心率150，运动诱发心绞痛0，ST段压低2.3，ST段斜率0，透视血管数0，地中海贫血1。最终诊断结果为："
    # 对文本进行编码
    input_ids = tokenizer.enc.encode_ordinary(sample_text)
    input_ids.insert(0, tokenizer.get_bos_token_id()) # 加上起始符
    tokens = [tokenizer.enc.decode([idx]) for idx in input_ids] # 获取对应的单词列表

    idx_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    B, T = idx_tensor.size()

    # 3. 加载刚才保存的最佳模型 (假设 Fold 0 表现最好)
    config = build_model_config(DEPTH)
    model = GPT(config).to(device)
    model.load_state_dict(torch.load("saved_models/model_fold0.pt", map_location=device))
    model.eval()

    # 4. 手动前向传播，拦截最后一层的 Q 和 K
    with torch.no_grad():
        cos_sin = model.cos[:, :T], model.sin[:, :T]
        x = model.transformer.wte(idx_tensor)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))

        x0 = x
        # 遍历前面的层
        for i in range(len(model.transformer.h) - 1):
            block = model.transformer.h[i]
            x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
            ve = model.value_embeds[str(i)](idx_tensor) if str(i) in model.value_embeds else None
            x = block(x, ve, cos_sin, model.window_sizes[i])

        # 截取最后一层
        i = len(model.transformer.h) - 1
        block = model.transformer.h[i]
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0

        x_norm = torch.nn.functional.rms_norm(x, (x.size(-1),))

        # ---- 替换开始 ----
        # 获取 Q 和 K，包含所有 Head
        q = block.attn.c_q(x_norm).view(B, T, block.attn.n_head, block.attn.head_dim)
        k = block.attn.c_k(x_norm).view(B, T, block.attn.n_kv_head, block.attn.head_dim)

        # 应用旋转位置编码 (RoPE)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        # 为了计算多头注意力，调整维度顺序 (B, n_head, T, head_dim)
        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)

        # 计算所有 Head 的注意力得分 Q * K^T / sqrt(d)
        attn_scores = (q_heads @ k_heads.transpose(-2, -1)) / (block.attn.head_dim ** 0.5)

        # 因果掩码 (Causal Mask)
        mask = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        # 将 mask 扩展以匹配多头维度 (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(~mask.view(1, 1, T, T), float('-inf'))

        # Softmax 得到权重，shape: (B, n_head, T, T)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # 将所有 Head 的权重取平均！这就是消除噪音的关键！
        # shape 变成 (B, T, T) -> 提取第一个 Batch -> (T, T)
        avg_attn_weights = attn_weights.mean(dim=1).squeeze(0).cpu().numpy()
        # ---- 替换结束 ----

    # 下方提取最后一行时，变量名改为 avg_attn_weights
    final_step_attention = avg_attn_weights[-1, 1:]
    tokens_to_plot = tokens[1:]
    # 6. 使用 Seaborn 绘制热力图
    # 把 figsize 的宽度从 14 改成 24，高度改成 3
    plt.figure(figsize=(24, 3))
    sns.heatmap([final_step_attention],
                xticklabels=tokens_to_plot,
                yticklabels=["Attention\nWeight"],
                cmap="Reds",
                cbar_kws={"orientation": "horizontal", "pad": 0.4})

    plt.title("Transformer Attention Weights on Clinical Features", fontsize=16)
    # 把字体稍微调小一点，角度倾斜到 60 度，防止重叠
    plt.xticks(rotation=60, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=300)
    print("热力图已成功保存为 attention_heatmap.png！")

if __name__ == "__main__":
    plot_attention_heatmap()
