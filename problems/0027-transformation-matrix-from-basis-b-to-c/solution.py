def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    import numpy as np
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    P = np.linalg.solve(C,B)
    return P