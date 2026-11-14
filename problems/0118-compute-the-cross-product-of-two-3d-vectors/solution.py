import numpy as np

def cross_product(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    ans = [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    return ans