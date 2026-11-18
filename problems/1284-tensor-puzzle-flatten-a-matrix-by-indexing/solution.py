import numpy as np

def flatten(A: np.ndarray) -> np.ndarray:
    """Row-major flatten of 2-D array A without reshape/ravel."""
    indexes = np.arange(A.shape[0]*A.shape[1])
    rows = indexes//A.shape[1]
    cols = indexes % A.shape[1]
    return A[rows,cols]