import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    """
    Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

    :param dense_matrix: 2D list representing a dense matrix
    :return: A tuple containing (values array, column indices array, row pointer array)
    """
    A = np.asarray(dense_matrix)
    rows = A.shape[0]
    cols = A.shape[1]
    vals = []
    col_indices = []
    row__indices = [0]
    ele = 0
    for i in range(rows):
        for j in range(cols):
            if A[i][j]!=0:
                vals.append(A[i][j])
                col_indices.append(j)
                ele = ele+1
        row__indices.append(ele)
    return (vals, col_indices, row__indices)
