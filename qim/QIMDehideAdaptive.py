import numpy as np
from scipy.fftpack import dct

from .Quantificate import quantificate

def QIMDehideAdaptive(stg, length):
    """ Extracts hidden information from a stego image using Quantization Index Modulation (QIM).
    
    Steps:
    1. Divides the image into 8x8 blocks.
    2. Applies the Discrete Cosine Transform (DCT) to each block.
    3. Quantizes the diagonal coefficients to extract the hidden information.
    4. Returns extracted information as a binary array of the specified length.

    Parameters:
        @param stg (numpy.ndarray): The stego image from which hidden information is to be extracted.
        @param length (int): The length of the hidden information to be extracted.
    
    Returns:
        @return numpy.ndarray: A binary array representing the extracted hidden information.
    
    """
    block = (8, 8)
    si = stg.shape
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
            tmp = stg[rst:red, cst:ced]
            tmp = dct(dct(tmp.T, norm='ortho').T, norm='ortho')
            
            coef = np.zeros(block[0])
            for k in range(block[0]):
                l = block[0] - 1 - k  # Position for diagonal coefficients
                coef[l] = tmp[k, l]
            
            delta = round(100 * np.mean(np.abs(coef))) / 10.0
            delta = max(delta, 15.5)
            
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