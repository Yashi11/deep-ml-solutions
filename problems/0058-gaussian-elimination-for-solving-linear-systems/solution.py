import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """

    total_rows = A.shape[0]
    total_cols = A.shape[1]
    for i in range(total_rows-1):
        pivot = i + np.argmax(np.abs(A[i:, i]))

        # Swap the pivot row into position
        A[[i, pivot]] = A[[pivot, i]]
        b[[i, pivot]] = b[[pivot, i]]
        if A[i][i]==0:
                raise ValueError
        for j in range(i+1, total_rows):
            factor = A[j][i]/A[i][i]
            A[j]=A[j]-(factor)*A[i]
            b[j] = b[j] - factor*b[i]
    i = total_rows-1
    while i>=0:
        for j in range(i+1,total_rows):
                b[i] = b[i]-A[i][j]*b[j]
        if A[i][i]==0:
                raise ValueError
        b[i] = b[i]/A[i][i]
        i = i-1
    return b
