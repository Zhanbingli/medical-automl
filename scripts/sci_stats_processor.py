import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def process_sci_results_v2(json_file):
    # 1. 加载数据
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {json_file}")
        return

    comparisons = data['comparisons']
    trans_summary = data['transformer_summary']

    # --- 数据准备 ---
    # 定义绘图顺序：Proposed 永远在第一个
    baseline_names = list(comparisons.keys())
    model_names = ['Proposed'] + baseline_names

    means = [trans_summary['mean_auc']] + [comparisons[m]['baseline_mean'] for m in baseline_names]
    stds = [trans_summary['std_auc']] + [np.std(comparisons[m]['baseline_folds']) for m in baseline_names]

    # 2. 绘制 SCI 风格层级显著性对比图 (Advanced Fusion Bar Plot)
    plt.figure(figsize=(12, 7)) # 稍微加宽，给标注留空间
    sns.set_style("white")

    # 配色设置：拟议模型使用深红色，基准模型使用渐变蓝色
    colors = ['#B22222'] + sns.color_palette("Blues_d", len(comparisons))

    # 绘制柱状图，并捕获返回值（用于获取坐标）
    x_positions = np.arange(len(model_names))
    width = 0.8
    bars = plt.bar(x_positions, means, yerr=stds, color=colors, width=width,
                   capsize=7, edgecolor='black', linewidth=1.2, alpha=0.9, zorder=2)

    # 3. 核心改进：精确对准与层级防碰撞标注

    # 3.1 获取每个柱子的精确中心 X 坐标
    # Matplotlib 的 bar.get_x() 返回的是柱子左边缘坐标
    bar_centers = [bar.get_x() + bar.get_width() / 2.0 for bar in bars]
    proposed_center_x = bar_centers[0] # Proposed 的中心 X 坐标

    # 3.2 动态确定线条高度 (层级对准算法)
    ymax_current = max([m + s for m, s in zip(means, stds)]) # 初始化最高点：所有柱子+误差棒的最大值
    step_height = 0.05  # 每层显著性线条之间的固定安全距离

    significant_count = 0 # 用于动态拉高绘图区域

    # 遍历所有基准模型进行标注
    for i, model_name in enumerate(baseline_names, 1):
        info = comparisons[model_name]
        p_val = info['p_value']
        sig_marker = info['significance']

        # 只有显著的（p < 0.05）才画线
        if p_val < 0.05:
            current_bar_x = bar_centers[i] # 当前基准模型的中心 X 坐标

            # 核心：计算新线条的 Y 轴高度。它必须比它下方的所有东西都高
            # 我们只需要比较 Proposed 柱子、当前柱子以及已经绘制的最高线条即可
            y_base = max(trans_summary['mean_auc'] + trans_summary['std_auc'],
                         info['baseline_mean'] + np.std(info['baseline_folds']),
                         ymax_current)

            y_line = y_base + step_height # 新线条的高度
            ymax_current = y_line + 0.02 # 更新当前的系统最高点（给星号留点空间）

            # 绘制精确对准的对比连线 (Brackets)
            x1, x2 = proposed_center_x, current_bar_x
            h = 0.012 # 钩子的高度

            # 绘制连线：起点钩子 -> 横线 -> 终点钩子
            plt.plot([x1, x1, x2, x2], [y_line-h, y_line, y_line, y_line-h],
                     lw=1.5, c='black', zorder=3)

            # 标注星号 (* 或 **)， ha='center' 确保文字中心与线条中心对齐
            plt.text((x1+x2)*0.5, y_line + 0.005, sig_marker,
                     ha='center', va='bottom', color='black',
                     fontweight='bold', fontsize=14, zorder=4)

            significant_count += 1

    # 4. 图表修饰
    # 动态调整 Y 轴上限，确保所有标注都在可见区域内
    plt.ylim(0.5, ymax_current + 0.05)

    plt.ylabel('Mean AUC Score', fontsize=13, fontweight='bold')
    plt.title('Performance Comparison with Statistical Significance', fontsize=15, pad=30)

    # 精确设置 X 轴刻度标签位置在柱子中心
    plt.xticks(x_positions, model_names, rotation=45, ha='right', fontsize=11)

    # 移除冗余边框 (SCI 常用样式)
    sns.despine()

    # 添加网格线增强可读性（可选）
    plt.grid(axis='y', linestyle=':', alpha=0.5, zorder=1)

    plt.tight_layout()
    output_filename = 'statistical_significance_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"✅ 修正后的显著性对比图已保存至: {output_filename}")

if __name__ == "__main__":
    process_sci_results_v2('statistical_tests_results.json')
