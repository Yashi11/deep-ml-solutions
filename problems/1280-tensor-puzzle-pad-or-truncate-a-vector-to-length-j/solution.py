import numpy as np

def pad_to(a: np.ndarray, j: int) -> np.ndarray:
    """Pad a with zeros (or truncate) to length j."""
    n = len(a)
    if j<=n:
        return a[:j]
    else:
        first = a
        second = np.zeros(j-n)
        return np.concatenate([first,second])
