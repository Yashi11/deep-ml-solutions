import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    n = len(a)
    index = np.arange(n,0,-1)-1
    return a[index]
