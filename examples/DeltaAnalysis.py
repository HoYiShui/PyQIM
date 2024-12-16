import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from .DeltaAnalysis_sample import watermark_analysis_sample

def plot_metrics(metrics_dict, output_dir=None):
    """
    Plot the changes of different metrics with delta using line charts of different colors, and optionally save the images.
    
    Parameters:
        @param metrics_dict (dict): 
            Dictionary containing multiple sets of data, where the key is the metric name (str) and the value is a tuple containing (delta_values, metric_values).
        @param output_dir (str): Optional, the folder path to save the images. If None, the images will not be saved.
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

    # Save the image locally
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "metrics_plot_vanilla_deltaVary.png")
        plt.savefig(output_path)
        print(f"Image saved to: {output_path}")
    
    plt.show()


def analyze_deltas_and_save(delta_values, noise_dB, output_dir=None):
    """
    Analyze the impact of delta changes on multiple metrics and save the data and plot results.
    
    Parameters:
        @param delta_values (numpy.ndarray): The range of delta values.
        @param noise_dB (float): Noise intensity.
        @param output_dir (str): Optional, the folder path to save the results. If None, the results will not be saved.
    """
    # Initialize lists to store results
    transparency_values = []
    corr_no_noise_values = []
    corr_with_noise_values = []
    similarity_with_noise_values = []

    # Iterate over delta_values and call the watermark_analysis function
    for delta in tqdm(delta_values, desc="Processing deltas", unit="step"):
        results = watermark_analysis_sample(delta, noise_dB)
        transparency_values.append(results[0])
        corr_no_noise_values.append(results[1])
        corr_with_noise_values.append(results[2])
        similarity_with_noise_values.append(results[3])

    # Save data to DataFrame
    results_df = pd.DataFrame({
        'Delta': delta_values,
        'PSNR': transparency_values,
        'Correlation (No Noise)': corr_no_noise_values,
        'Correlation (With Noise)': corr_with_noise_values,
        'Similarity (With Noise)': similarity_with_noise_values
    })

    # Save data as CSV file
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "metrics_results_vanilla_deltaVary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Data saved to: {csv_path}")

    # Prepare data for plotting
    metrics_dict = {
        'PSNR': (delta_values, transparency_values),
        'Correlation (No Noise)': (delta_values, corr_no_noise_values),
        'Correlation (With Noise)': (delta_values, corr_with_noise_values),
        'Similarity (With Noise)': (delta_values, similarity_with_noise_values)
    }

    # Plot and save the figures
    plot_metrics(metrics_dict, output_dir)


if __name__ == '__main__':
    delta_values = np.arange(10, 100.5, 0.5)
    noise_dB_example = 35
    output_dir = "."
    analyze_deltas_and_save(delta_values, noise_dB_example, output_dir)
