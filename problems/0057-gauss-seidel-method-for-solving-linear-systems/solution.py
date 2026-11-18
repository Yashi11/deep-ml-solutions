import numpy as np

def LU_sum_decomposition(A):
    tmp = np.arange(A.shape[0])
    rows = tmp[:,None]
    cols = tmp[None,:]
    mask_l = rows>=cols
    mask_u = rows<cols
    L = np.where(mask_l, A, 0)
    U = np.where(mask_u, A, 0)
    return (L,U)

def gauss_seidel(A, b, n, x_ini=None):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    L, U = LU_sum_decomposition(A)
    if x_ini is None:
        x = np.zeros_like(b)
    else:
        x = x_ini
    while(n):
        x_new = np.linalg.inv(L)@(b - U@x)
        x = x_new
        n = n -1
    return x
