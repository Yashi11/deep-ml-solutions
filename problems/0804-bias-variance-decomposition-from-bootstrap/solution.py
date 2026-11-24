import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions,dtype=float)
    y_true = np.asarray(y_true,dtype=float)
    mean_prediction = np.mean(predictions,axis=0)
    squared_bias = (y_true - mean_prediction)**2
    variance = np.mean((predictions-mean_prediction)**2,axis=0)
    mse = np.mean((predictions-y_true)**2)

    return {'bias_squared': float(np.mean(squared_bias)), 'variance': float(np.mean(variance)), 'mse': float(mse)}