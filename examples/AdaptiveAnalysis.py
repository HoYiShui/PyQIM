import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from qim.utils.Plot import plot_metrics
from .AdaptiveAnalysis_sample import watermark_analysis_sample

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

    # Iterate over noise_dB_values and call the watermark_analysis function
    for noise_dB in tqdm(noise_dB_values, desc="Processing noise levels", unit="step"):
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

    # Prepare data for plotting
    metrics_dict = {
        'PSNR': (noise_dB_values, transparency_values),
        'Correlation (No Noise)': (noise_dB_values, corr_no_noise_values),
        'Correlation (With Noise)': (noise_dB_values, corr_with_noise_values),
        'Similarity (With Noise)': (noise_dB_values, similarity_with_noise_values)
    }

    if output_dir:
        # Save data as CSV file
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "metrics_results_adaptive_noiseVary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Data saved to: {csv_path}")

        # Plot and save the figures
        plot_path = os.path.join(output_dir, "metrics_plot_adaptive_noiseVary.png")
        plot_metrics('Noise_dB', metrics_dict, plot_path)
    else:
        # Plot without saving
        plot_metrics('Noise_dB', metrics_dict)

if __name__ == '__main__':
    noise_dB_values = np.arange(15, 55, 0.5)
    output_dir = "./examples/output"
    analyze_deltas_and_save(noise_dB_values, output_dir)