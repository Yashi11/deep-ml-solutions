import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    # Soln without numpy broadcasting
    sums = []
    counts = np.asarray(counts,dtype=float)
    prob = counts.copy()
    rows = counts.shape[0]
    cols = counts.shape[1]
    for i in range(rows):
        sum = 0.0
        for j in range(cols):
            sum = sum+counts[i][j]
        sums.append(sum)
    for i in range(rows):
        for j in range(cols):
            if sums[i]==0:
                prob[i][j]=0.0
            else:
                prob[i][j]=prob[i][j]/sums[i]
    return prob
