import numpy as np
def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true (float | list[float]): Ground truth values
        y_pred (float | list[float]): Predicted values
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    # Your code here
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = np.abs(y_true-y_pred) <= delta
    opt1 = 0.5*((y_true-y_pred)**2)
    opt2 = delta*(np.abs(y_true-y_pred)-0.5*delta)
    hubber_array = np.where(mask,opt1,opt2)
    return np.mean(hubber_array)