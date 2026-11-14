import numpy as np

def make_diagonal(x):
    # Your code here
    x = np.asarray(x)
    n = x.shape[0]
    I = np.identity(n,dtype=float)
    return x*I