import numpy as np
from scipy.fftpack import dct

from .Quantificate import quantificate

def QIMDehide(I, delta, length):
    """ Extracts embedded data from an image using Quantization Index Modulation (QIM).

    Notes:
    - The image is divided into non-overlapping 8x8 blocks.
    - The data is extracted from the diagonal coefficients of the DCT-transformed blocks.
    - The function uses the Discrete Cosine Transform (DCT) and its inverse (IDCT) for extraction.
    
    Parameters:
        @param I (numpy.ndarray): The input image from which data will be extracted.
        @param delta (float): The quantization step size used for embedding.
        @param length (int): The length of the binary data to be extracted.
    
    Returns:
        @return numpy.ndarray: The extracted binary data.
    
    """
    block = (8, 8)
    si = I.shape
    N = si[1] // block[1]  # Number of blocks in each row
    M = si[0] // block[0]  # Number of blocks in each column
    o = np.zeros(M * N, dtype=int)
    idx = 0
    
    for i in range(M):
        rst = i * block[0]
        red = (i + 1) * block[0]
        
        for j in range(N):
            cst = j * block[1]
            ced = (j + 1) * block[1]
            tmp = I[rst:red, cst:ced]
            tmp = dct(dct(tmp.T, norm='ortho').T, norm='ortho')
            
            to = np.zeros(block[0], dtype=int)
            for k in range(block[0] - 1, -1, -1):
                l = block[0] - 1 - k  # Position for diagonal coefficients
                q00 = quantificate(tmp[k, l], 0, delta)
                q10 = quantificate(tmp[k, l], 1, delta)
                pos = np.argmin(np.abs(tmp[k, l] - np.array([q00, q10])))
                to[l] = pos
            
            if np.sum(to) >= 4:
                o[idx] = 1
            else:
                o[idx] = 0
            idx += 1
    
    return o[:length]