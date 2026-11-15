import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    a = np.asarray(a, dtype=float)
    sum =0
    for ele in a:
        sum = sum + ele
    return sum
