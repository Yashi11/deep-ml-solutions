def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []
    total_rows = len(matrix)
    total_cols = len(matrix[0])
    if mode == 'row':
        for i in range(total_rows):
            mean = 0.0
            for j in range(total_cols):
                mean = mean+matrix[i][j]
            means.append(mean/total_cols)
    elif mode == 'column':
        for i in range(total_cols):
            mean = 0.0
            for j in range(total_rows):
                mean = mean+matrix[j][i]
            means.append(mean/total_rows)
    return means