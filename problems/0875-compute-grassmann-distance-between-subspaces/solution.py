import numpy as np

def grassmann_distance(U_A: list[list[float]], U_B: list[list[float]]) -> float:
    """
    Compute the projection-based Grassmann distance between two subspaces
    represented by column-orthonormal matrices U_A and U_B.
    """
    U_A = np.asarray(U_A, dtype=float)
    U_B = np.asarray(U_B, dtype=float)
    p = min(U_A.shape[1],U_B.shape[1])
    cross_product = U_A.T@U_B
    singular_values = np.linalg.svd(cross_product, compute_uv=False)
    sigma_singular = np.sum(singular_values@singular_values)
    return np.sqrt(max(p-sigma_singular,0))