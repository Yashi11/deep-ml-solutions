import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    # Your code here
    A = np.asarray(A, dtype=float)
    A_t = A.T
    Q = []
    n = A_t.shape[0]
    for i in range(n):
        q = A_t[i].copy()
        for j in range(i):
            q = q - (np.dot(A_t[i],Q[j])*Q[j])
        length = np.linalg.norm(q)
        if np.isclose(length,0):
            raise ValueError
        Q.append(q/length)

    R = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(i,n):
           R[i][j]=np.dot(A_t[j],Q[i])
    
    Q = np.column_stack(Q)
    return (Q.tolist(),R.tolist())
