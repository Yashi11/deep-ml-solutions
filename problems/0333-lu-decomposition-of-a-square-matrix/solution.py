import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.
    
    Args:
        A: Square matrix as a list of lists
    
    Returns:
        tuple: (L, U) where L is lower triangular with 1s on diagonal,
               U is upper triangular, and A = L @ U
    """
    A = np.asarray(A,dtype=float)
    total_rows = A.shape[0]
    L = np.identity(total_rows)
    U = A.copy()
    for i in range(total_rows):
        if U[i][i]==0:
            raise ValueError
        for j in range(i+1,total_rows):
            factor = U[j][i]/U[i][i]
            U[j]=U[j]-factor*U[i]
            L[j][i]=factor
    return (L,U)