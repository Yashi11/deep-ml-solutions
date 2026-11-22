
import numpy as np

def rmse(y_true, y_pred):
    # Write your code here
    if y_pred.size==0 or y_true.size==0:
        raise ValueError
    if y_pred.shape != y_true.shape:
        raise ValueError
    rmse_res = np.sqrt(np.mean((y_true-y_pred)**2))
    return round(rmse_res,3)
