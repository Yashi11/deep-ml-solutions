import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    u = np.arange(n)
    mask = u[:,None]<=u[None,:]
    return mask.astype(float)
