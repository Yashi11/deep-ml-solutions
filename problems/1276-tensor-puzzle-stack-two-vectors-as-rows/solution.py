import numpy as np

def vstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stack 1-D arrays a and b as rows of a (2, n) matrix."""
    # return np.block([[a],[b]])
    rows = np.arange(2).reshape(-1, 1)
    return np.where(rows == 0, a, b)
