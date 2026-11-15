import numpy as np

def ones(n: int) -> np.ndarray:
    """Return a length-n float vector of ones without calling np.ones."""
    arr = []
    for i in range(n):
        arr.append(1.0)
    return arr