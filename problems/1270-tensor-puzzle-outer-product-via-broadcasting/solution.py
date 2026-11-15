import numpy as np

def outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer product of 1-D arrays a and b via broadcasting."""
    a = np.asarray(a,dtype=float)
    b = np.asarray(b,dtype=float)
    ans = []
    for ele in a:
        tmp = np.arange(len(b))*0+ele
        ans.append((b*tmp).tolist())
    return ans
