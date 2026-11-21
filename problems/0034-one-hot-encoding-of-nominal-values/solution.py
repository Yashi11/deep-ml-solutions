import numpy as np

def to_categorical(x, n_col=None):
    # Your code here
    x = np.asarray(x)
    if n_col is None:
        n_col = int(np.max(x)) + 1 if x.size else 0
    tmp = np.arange(n_col)
    return np.where(x[:, None] == tmp[None, :], 1.0, 0.0)    