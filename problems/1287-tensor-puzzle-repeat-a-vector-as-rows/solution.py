import numpy as np

def repeat_rows(a: np.ndarray, d: int) -> np.ndarray:
    """Stack d copies of 1-D array a as rows."""
    b = np.zeros(d).reshape(d,-1)
    return a+b 
