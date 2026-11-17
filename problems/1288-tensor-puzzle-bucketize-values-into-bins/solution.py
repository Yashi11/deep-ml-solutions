import numpy as np

def bucketize(v: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return bucket indices for v given sorted boundaries."""
    i = 0
    b = 0
    ans = []
    while i<len(v):
        if b>=len(boundaries):
            ans.append(b)
            i = i+1
        elif v[i]< boundaries[b]:
            ans.append(b)
            i = i+1
        else:
            b = b+1
    return ans
