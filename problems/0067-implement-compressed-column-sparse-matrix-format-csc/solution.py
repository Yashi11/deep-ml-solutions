import numpy as np
def compressed_col_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

    :param dense_matrix: List of lists representing the dense matrix
    :return: Tuple of (values, row indices, column pointer)
    """
    A = np.asarray(dense_matrix)
    rows = A.shape[0]
    cols = A.shape[1]
    vals = []
    row_indices = []
    col_indices = [0]
    ele = 0
    for i in range(cols):
        for j in range(rows):
            if A[j][i]!=0:
                vals.append(A[j][i])
                row_indices.append(j)
                ele = ele + 1
        col_indices.append(ele)
    return (vals,row_indices,col_indices)
