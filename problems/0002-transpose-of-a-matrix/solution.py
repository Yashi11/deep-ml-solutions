def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    m = len(a)
    n = len(a[0])
    ans = []
    for i in range(n):
        ans_i = []
        for j in range(m):
            ans_i.append(a[j][i])
        ans.append(ans_i)
    return ans
    pass