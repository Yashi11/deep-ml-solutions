import numpy as np

def roll(a: np.ndarray) -> np.ndarray:
    """Circular left shift by one."""
    return np.concatenate([a[1:],a[0:1]])
