import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from .DeltaAnalysis_sample import watermark_analysis_sample

def plot_metrics(metrics_dict, output_dir=None):
    """
    绘制不同指标随 delta 的变化，使用不同颜色的折线图，并可选择保存图像。
    
    Parameters:
        @param metrics_dict (dict): 
            包含多组数据的字典，键为指标名 (str)，值为包含 (delta_values, metric_values) 的 tuple。
        @param output_dir (str): 可选，保存图像的文件夹路径。如果为 None，则不保存。
    """
    colors = ['b', 'g', 'r', 'm']
    plt.figure(figsize=(12, 8))
    
    for i, (metric_name, (delta_values, metric_values)) in enumerate(metrics_dict.items(), start=1):
        plt.subplot(2, 2, i)
        plt.plot(delta_values, metric_values, color=colors[i % len(colors)], label=metric_name)
        plt.title(f'{metric_name} vs. Delta')
        plt.xlabel('Delta')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    # 保存图像到本地
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metrics_plot_vanilla_deltaVary.png")
        plt.savefig(output_path)
        print(f"图像已保存至: {output_path}")
    
    plt.show()


def analyze_deltas_and_save(delta_values, noise_dB, output_dir=None):
    """
    分析 delta 的变化对多个指标的影响，并保存数据与绘图结果。
    
    Parameters:
        @param delta_values (numpy.ndarray): delta 的取值范围。
        @param noise_dB (float): 噪声强度。
        @param output_dir (str): 可选，保存结果的文件夹路径。如果为 None，则不保存。
    """
    # 初始化存储结果的列表
    transparency_values = []
    corr_no_noise_values = []
    corr_with_noise_values = []
    similarity_with_noise_values = []

    # 遍历 delta_values，调用 watermark_analysis 函数
    for delta in tqdm(delta_values, desc="Processing deltas", unit="step"):
        results = watermark_analysis_sample(delta, noise_dB)
        transparency_values.append(results[0])
        corr_no_noise_values.append(results[1])
        corr_with_noise_values.append(results[2])
        similarity_with_noise_values.append(results[3])

    # 保存数据到 DataFrame
    results_df = pd.DataFrame({
        'Delta': delta_values,
        'PSNR': transparency_values,
        'Correlation (No Noise)': corr_no_noise_values,
        'Correlation (With Noise)': corr_with_noise_values,
        'Similarity (With Noise)': similarity_with_noise_values
    })

    # 保存数据为 CSV 文件
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "metrics_results_vanilla_deltaVary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"数据已保存至: {csv_path}")

    # 准备绘图数据
    metrics_dict = {
        'PSNR': (delta_values, transparency_values),
        'Correlation (No Noise)': (delta_values, corr_no_noise_values),
        'Correlation (With Noise)': (delta_values, corr_with_noise_values),
        'Similarity (With Noise)': (delta_values, similarity_with_noise_values)
    }

    # 绘制图形并保存
    plot_metrics(metrics_dict, output_dir)


if __name__ == '__main__':
    delta_values = np.arange(10, 100.5, 0.5)
    noise_dB_example = 35
    output_dir = "."
    analyze_deltas_and_save(delta_values, noise_dB_example, output_dir)
