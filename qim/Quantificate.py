import numpy as np

def quantificate(value, bit, delta):
    """ Quantizes a given value based on the provided bit and quantization step size (delta).
    
    Notes:
    - The function applies a dither based on the bit value to the input value before quantizing.
    - The dither is calculated to map bit values 0 and 1 to -1 and +1 respectively.
    - The quantization is performed by rounding the dithered value to the nearest multiple of delta.
    
    Parameters:
        @param value (float): The input value to be quantized.
        @param bit (int): The binary bit (0 or 1) used to determine the dither.
        @param delta (float): The quantization step size.
    
    Returns:
        @return float: The quantized value.
    
    """
    dither = (bit * 2 - 1) * (delta / 4)  # '(bit * 2 - 1)' makes '0,1' to '-1, +1'
    return delta * round((value - dither) / delta) + dither