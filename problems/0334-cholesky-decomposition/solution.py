import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    try:
        A = np.asarray(A,dtype=float)
    except (TypeError, ValueError):
        return -1
    if A.ndim!=2 or A.shape[0]==0:
        return -1

    n,m = A.shape

    if n!=m or not np.array_equal(A,A.T):
        return -1
    
    if not np.all(np.isfinite(A)):
        return -1

    L = np.zeros((n,n),dtype=float)

    for i in range(n):
        for j in range(i+1):
            tmp = 0
            for k in range(j):
                tmp = tmp+(L[i][k]*L[j][k])
            if i==j:
                tmp_2 = A[i][j]-tmp
                if tmp_2<=0:
                    return -1
                L[i][j] = np.sqrt(tmp_2)
            else:
                L[i][j]=(A[i][j]-tmp)/(L[j][j])
    return L.tolist()