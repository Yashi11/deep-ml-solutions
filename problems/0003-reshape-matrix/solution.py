import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    #Write your code here and return a python list after reshaping by using numpy's tolist() method
    org_rows = len(a)
    org_cols = len(a[0])
    new_rows = new_shape[0]
    new_cols = new_shape[1]
    if org_rows*org_cols == new_rows*new_cols:
        elements = []
        for i in range(org_rows):
            for j in range(org_cols):
                elements.append(a[i][j])
        reshaped_matrix=[]
        k = 0
        for i in range(new_rows):
            tmp = []
            for j in range(new_cols):
                tmp.append(elements[k])
                k = k+1
            reshaped_matrix.append(tmp)
        return reshaped_matrix
    else:
        return []