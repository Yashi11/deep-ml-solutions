import numpy as np

def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""
    n = len(a)
    # make lower triangular matrix using index comparision**
    tmp = np.arange(n)
    mask = tmp[:,None]>=tmp[None,:]
    l = mask.astype(int)
    nu = a.T*l
    ans = nu@(np.ones(n).reshape(-1,1))
    return ans.reshape(-1) # or ans.flatten()

