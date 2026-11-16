import numpy as np

def linspace(start, stop, n: int) -> np.ndarray:
    """n evenly spaced values from start to stop inclusive."""
    if n<2:
        if n==1:
            return [start]
        elif n==0:
            return []
        else:
            raise ValueError
    move = (stop-start)/(n-1)
    ans = []
    for i in range(n):
        ans.append(start)
        start=start+move
    return ans
