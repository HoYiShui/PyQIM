import numpy as np

def Similar(x, y):
    """Calculate the cosine similarity between two sequences.

    Notes:
        The cosine similarity is a measure of similarity between two non-zero vectors of an inner product space
        that measures the cosine of the angle between them. This function computes the cosine similarity
        between two input sequences x and y.

    Parameters:
        @param x (list or array-like): The first input sequence.
        @param y (list or array-like): The second input sequence.

    Returns:
        @return float: The cosine similarity between the two input sequences.

    Usage example:
        >>> Similar([1, 2, 3], [4, 5, 6])
        0.9746318461970762

    """
    len_min = min(len(x), len(y))
    x = np.array(x[:len_min], dtype=float)
    y = np.array(y[:len_min], dtype=float)
    return np.sum(x * y) / (np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2)))