import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    # x_new = x_old + D_inv(b-Ax_old)
    A = np.asarray(A, dtype=float)
    x = np.zeros_like(b)
    rows = A.shape[0]
    cols = A.shape[1]
    I = np.identity(rows)
    D = A*I
    for i in range(n):
        x_new = x + np.linalg.inv(D)@(b-A@x)
        x = x_new
    return np.round(x,4)