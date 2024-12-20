import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import random_noise

from qim.QIMHide import QIMHide
from qim.QIMDehide import QIMDehide
from qim.utils.Similar import Similar
from qim.utils.Plot import plot_images

base_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.abspath(os.path.join(base_dir, '../qim/data'))

carrier_path = os.path.join(data_dir, 'lena_512.bmp')
watermark_path = os.path.join(data_dir, 'logo_64.bmp')

# Initialization and Image Loading
I = cv2.imread(carrier_path, cv2.IMREAD_GRAYSCALE) # carrier_img
d = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)  # watermark_img
data = d.flatten()

# Embedding Watermark
delta = 25.5
stg = QIMHide(I, data, delta) # stego_img
transparency = psnr(I, stg)

imgs = {
    "Org": I,
    f'PSNR: {transparency:.2f}': stg
}
plot_images(imgs)

# Extracting Watermark without Noise
msg = QIMDehide(stg, delta, len(data)) # extracted watermark
m = msg.reshape((64, 64))
s = Similar(d, m)

imgs = {
    "watermark": d,
    f'{s:.2f}': m,
    'differ': np.logical_xor(d.astype(np.int32), m)
}
plot_images(imgs)

# Extracting Watermark with Noise
noise_dB = 35
noise_var = 10 ** (- noise_dB / 10)
y = (random_noise(stg / 255.0, mode='gaussian', var=noise_var) * 255).astype(stg.dtype)
transparency_noise = psnr(stg, y, data_range=stg.max() - stg.min())

imgs = {
    f'PSNR: {transparency:.2f}': stg,
    f'Gaussian: {transparency_noise:.2f}': y
}
plot_images(imgs)

# Extracting Watermark with Noise
msg = QIMDehide(y, delta, len(data))
m = msg.reshape((64, 64))
sn = Similar(d, m)

imgs = {
    'watermark': d,
    f'{sn:.2f}': m,
    'differ': np.logical_xor(d.astype(np.int32), m)
}
plot_images(imgs)