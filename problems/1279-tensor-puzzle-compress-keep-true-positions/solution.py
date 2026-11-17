import numpy as np

def compress(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack v[g] into the front of a zero vector of length len(v)."""
    n = len(g)
    indexes = np.where(g[:]==True)
    v_ = v[indexes]
    m = len(v_)
    return np.concatenate([v_,np.zeros(n-m)])

