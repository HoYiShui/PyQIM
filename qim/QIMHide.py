import numpy as np
from scipy.fftpack import dct, idct

from .Quantificate import quantificate

def QIMHide(I, data, delta):
    """ Embeds data into an image using Quantization Index Modulation (QIM).

    Notes:
    - The image is divided into non-overlapping 8x8 blocks.
    - The data is embedded into the diagonal coefficients of the DCT-transformed blocks.
    - If the length of the data is less than the number of blocks, the data is zero-padded.
    - The function uses the Discrete Cosine Transform (DCT) and its inverse (IDCT) for embedding.

    Parameters:
        @param I (numpy.ndarray): The input image in which data will be embedded.
        @param data (numpy.ndarray): The binary data to be embedded into the image.
        @param delta (float): The quantization step size used for embedding.
    
    Returns:
        @return numpy.ndarray: The image with embedded data.
    """
    block = (8, 8)
    si = I.shape
    lend = len(data)
    N = si[1] // block[1]  # Number of blocks in each row
    M = min(si[0] // block[0], -(-lend // N))  # Number of blocks in each column (ceil(lend / N))
    
    if lend < M * N:
        data = np.concatenate([data, np.zeros(M * N - lend)])
    
    o = I.copy()
    idx = 0
    
    for i in range(M):
        rst = i * block[0]
        red = (i + 1) * block[0]
        
        for j in range(N):
            cst = j * block[1]
            ced = (j + 1) * block[1]
            tmp = I[rst:red, cst:ced]
            tmp = dct(dct(tmp.T, norm='ortho').T, norm='ortho')
            
            for k in range(block[0] - 1, -1, -1):
                l = block[0] - 1 - k  # Position for diagonal coefficients
                tmp[k, l] = quantificate(tmp[k, l], data[idx], delta)
            
            o[rst:red, cst:ced] = idct(idct(tmp.T, norm='ortho').T, norm='ortho')
            idx += 1
    
    return o