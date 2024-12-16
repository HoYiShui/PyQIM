import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from .AdaptiveAnalysis_sample import watermark_analysis_sample

def plot_metrics(metrics_dict, output_dir=None):
    """
    Plot the changes of different metrics with noise using line charts of different colors, and optionally save the images.
    
    Parameters:
        @param metrics_dict (dict): 
            Dictionary containing multiple sets of data, where the key is the metric name (str) and the value is a tuple containing (noise_values, metric_values).
        @param output_dir (str): Optional, the folder path to save the images. If None, the images will not be saved.
    """
    colors = ['b', 'g', 'r', 'm']
    plt.figure(figsize=(12, 8))
    
    for i, (metric_name, (noise_dB_values, metric_values)) in enumerate(metrics_dict.items(), start=1):
        plt.subplot(2, 2, i)
        plt.plot(noise_dB_values, metric_values, color=colors[i % len(colors)], label=metric_name)
        plt.title(f'{metric_name} vs. Noise dB')
        plt.xlabel('Noise dB')
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.legend()

    plt.tight_layout()

    # Save the image locally
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metrics_plot_adaptive_noiseVary.png")
        plt.savefig(output_path)
        print(f"Image saved to: {output_path}")
    
    plt.show()


def analyze_deltas_and_save(noise_dB_values, output_dir=None):
    """
    Analyze the impact of noise changes on multiple metrics and save the data and plot results.
    
    Parameters:
        @param noise_dB_values (numpy.ndarray): The range of noise values.
        @param output_dir (str): Optional, the folder path to save the results. If None, the results will not be saved.
    """
    # Initialize lists to store results
    transparency_values = []
    corr_no_noise_values = []
    corr_with_noise_values = []
    similarity_with_noise_values = []

    # Iterate over delta_values and call the watermark_analysis function
    for noise_dB in tqdm(noise_dB_values, desc="Processing deltas", unit="step"):
        results = watermark_analysis_sample(noise_dB)
        transparency_values.append(results[0])
        corr_no_noise_values.append(results[1])
        corr_with_noise_values.append(results[2])
        similarity_with_noise_values.append(results[3])

    # Save data to DataFrame
    results_df = pd.DataFrame({
        'Noise_dB': noise_dB_values,
        'PSNR': transparency_values,
        'Correlation (No Noise)': corr_no_noise_values,
        'Correlation (With Noise)': corr_with_noise_values,
        'Similarity (With Noise)': similarity_with_noise_values
    })

    # Save data as CSV file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "metrics_results_adaptive_noiseVary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Data saved to: {csv_path}")

    # Prepare data for plotting
    metrics_dict = {
        'PSNR': (noise_dB_values, transparency_values),
        'Correlation (No Noise)': (noise_dB_values, corr_no_noise_values),
        'Correlation (With Noise)': (noise_dB_values, corr_with_noise_values),
        'Similarity (With Noise)': (noise_dB_values, similarity_with_noise_values)
    }

    # Plot and save the figures
    plot_metrics(metrics_dict, output_dir)


if __name__ == '__main__':
    noise_dB_example = np.arange(15, 55, 0.5)
    output_dir = "."
    analyze_deltas_and_save(noise_dB_example, output_dir)