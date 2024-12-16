import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import random_noise
from scipy.stats import pearsonr
from qim.QIMHide import QIMHide
from qim.QIMDehide import QIMDehide
from qim.utils.Similar import Similar

# Get paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, '../qim/data'))

carrier_path = os.path.join(data_dir, 'lena_512.bmp')
watermark_path = os.path.join(data_dir, 'logo_64.bmp')

# Load images
I = cv2.imread(carrier_path, cv2.IMREAD_GRAYSCALE)  # Carrier image
d = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)  # Watermark image
data = d.flatten()  # Flatten the watermark to one dimension

def watermark_analysis_sample(delta, noise_dB):
    """
    Function to perform watermark embedding, extraction, and quantitative analysis
    Parameters:
        @param delta: Quantization step size
        @param noise_dB: Noise intensity (in dB)
    Returns:
        @return: [PSNR, correlation coefficient without noise, correlation coefficient with noise, similarity with noise]
    """
    # Embed watermark
    stego_img = QIMHide(I, data, delta)

    # Calculate PSNR between the original image and the watermarked image
    transparency = psnr(I, stego_img)

    # Extract watermark without noise
    extracted_watermark = QIMDehide(stego_img, delta, len(data))
    extracted_watermark_img = extracted_watermark.reshape(d.shape)
    
    # Calculate correlation coefficient without noise
    corr_no_noise, _ = pearsonr(data, extracted_watermark.flatten())

    # Add noise
    noise_var = 10 ** (-noise_dB / 10)  # Convert noise intensity from dB to variance
    noisy_stego_img = (random_noise(stego_img / 255.0, mode='gaussian', var=noise_var) * 255).astype(stego_img.dtype)

    # Calculate PSNR of the noisy image
    transparency_with_noise = psnr(stego_img, noisy_stego_img, data_range=stego_img.max() - stego_img.min())

    # Extract watermark under noisy conditions
    extracted_noisy_watermark = QIMDehide(noisy_stego_img, delta, len(data))
    extracted_noisy_watermark_img = extracted_noisy_watermark.reshape(d.shape)

    # Calculate correlation coefficient with noise
    corr_with_noise, _ = pearsonr(data, extracted_noisy_watermark.flatten())

    # Calculate similarity with noise
    similarity_with_noise = Similar(d, extracted_noisy_watermark_img)

    # Return results
    return [transparency, corr_no_noise, corr_with_noise, similarity_with_noise]

if __name__ == '__main__':
    delta_example = 25.5
    noise_dB_example = 35
    results = watermark_analysis_sample(delta_example, noise_dB_example)
    print(f"Results for delta={delta_example}, noise_dB={noise_dB_example}: {results}")