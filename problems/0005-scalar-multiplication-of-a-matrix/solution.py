def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    # Your code here
    import numpy as np

    ans = np.asarray(matrix).copy()
    for i in range(ans.shape[0]):
        for j in range(ans.shape[1]):
            ans[i][j] = ans[i][j]* scalar
    return ans.tolist()