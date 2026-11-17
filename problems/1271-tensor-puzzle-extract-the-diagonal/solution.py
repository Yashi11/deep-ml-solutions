import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    A = np.asarray(A)
    n = A.shape[0]
    b = np.arange(n)
    return A[b,b]
