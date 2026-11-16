import numpy as np

def heaviside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise heaviside with zero-value b."""
    mask1 = np.asarray(a==0,dtype=int)
    ans1 = mask1*b
    ans2 = np.asarray(a>0, dtype=int)
    return (ans1+ans2)

