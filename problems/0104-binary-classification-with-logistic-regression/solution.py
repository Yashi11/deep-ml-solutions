import numpy as np
import math

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Your code here
    y_pred = X@weights + bias
    sigm_y_pred = np.array([1/(1+(math.e**(-1*val))) for val in y_pred])
    mask = sigm_y_pred>=0.5
    result = np.where(mask,1,0)

    return result