import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import random_noise
from scipy.stats import pearsonr
from qim.QIMHide import QIMHide
from qim.QIMDehide import QIMDehide
from qim.utils.Similar import Similar

# 获取路径
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(base_dir, '../qim/data'))

carrier_path = os.path.join(data_dir, 'lena_512.bmp')
watermark_path = os.path.join(data_dir, 'logo_64.bmp')

# 加载图像
I = cv2.imread(carrier_path, cv2.IMREAD_GRAYSCALE)  # 载体图像
d = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)  # 水印图像
data = d.flatten()  # 将水印展开为一维

def watermark_analysis_sample(delta, noise_dB):
    """
    执行加水印、提取水印以及量化分析的函数
    :param delta: 量化步长
    :param noise_dB: 噪声强度（以 dB 表示）
    :return: [PSNR, 无噪声相关系数, 有噪声相关系数, 有噪声相似度]
    """
    # 加水印
    stego_img = QIMHide(I, data, delta)

    # 计算原图和含水印图像的 PSNR
    transparency = psnr(I, stego_img)

    # 无噪声条件下提取水印
    extracted_watermark = QIMDehide(stego_img, delta, len(data))
    extracted_watermark_img = extracted_watermark.reshape(d.shape)
    
    # 计算无噪声下的相关系数
    corr_no_noise, _ = pearsonr(data, extracted_watermark.flatten())

    # 添加噪声
    noise_var = 10 ** (-noise_dB / 10)  # 将噪声强度从 dB 转换为方差
    noisy_stego_img = (random_noise(stego_img / 255.0, mode='gaussian', var=noise_var) * 255).astype(stego_img.dtype)

    # 计算含噪声图像的 PSNR
    transparency_with_noise = psnr(stego_img, noisy_stego_img, data_range=stego_img.max() - stego_img.min())

    # 含噪声条件下提取水印
    extracted_noisy_watermark = QIMDehide(noisy_stego_img, delta, len(data))
    extracted_noisy_watermark_img = extracted_noisy_watermark.reshape(d.shape)

    # 计算含噪声下的相关系数
    corr_with_noise, _ = pearsonr(data, extracted_noisy_watermark.flatten())

    # 计算含噪声下的相似度
    similarity_with_noise = Similar(d, extracted_noisy_watermark_img)

    # 返回结果
    return [transparency, corr_no_noise, corr_with_noise, similarity_with_noise]

if __name__ == '__main__':
    delta_example = 25.5
    noise_dB_example = 35
    results = watermark_analysis_sample(delta_example, noise_dB_example)
    print(f"Results for delta={delta_example}, noise_dB={noise_dB_example}: {results}")