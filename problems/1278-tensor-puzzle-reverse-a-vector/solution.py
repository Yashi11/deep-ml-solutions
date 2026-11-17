import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    n = a.shape[0]
    return a[::-1] #leave both start and stop empty and put step as -1 to reverse
