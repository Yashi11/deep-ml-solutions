import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    i = np.arange(n)
    mask = i[:,None]==i[None,]
    I = mask.astype(float)
    return I
