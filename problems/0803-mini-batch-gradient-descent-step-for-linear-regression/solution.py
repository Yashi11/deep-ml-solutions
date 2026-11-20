import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m=len(batch_indices)

    mini_batch_x = X[batch_indices]
    mini_batch_y = y[batch_indices]

    if m==0:
        raise ValueError

    Lw = 2*mini_batch_x.T@(mini_batch_x@weights + bias -mini_batch_y)*(1/m)
    Lb = 2*np.sum(mini_batch_x@weights + bias - mini_batch_y)*(1/m)

    weights = weights - lr*Lw
    bias = bias - lr*Lb

    return np.append(weights,bias)

