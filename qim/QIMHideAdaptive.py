import numpy as np
from scipy.fftpack import dct, idct

from .Quantificate import quantificate

def QIMHideAdaptive(I, data):
    block = (8, 8)
    si = I.shape
    lend = len(data)
    N = si[1] // block[1]  # Number of blocks in each row
    M = min(si[0] // block[0], -(-lend // N))  # Number of blocks in each column

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

            coef = np.zeros(block[0])
            for k in range(block[0]):
                l = block[0] - 1 - k  # Position for diagonal coefficients
                coef[l] = tmp[k, l]

            delta = round(100 * np.mean(np.abs(coef))) / 10.0
            delta = max(delta, 15.5)

            for k in range(block[0] - 1, -1, -1):
                l = block[0] - 1 - k  # Position for diagonal coefficients
                tmp[k, l] = quantificate(tmp[k, l], data[idx], delta)
            
            o[rst:red, cst:ced] = idct(idct(tmp.T, norm='ortho').T, norm='ortho')
            idx += 1

    return o