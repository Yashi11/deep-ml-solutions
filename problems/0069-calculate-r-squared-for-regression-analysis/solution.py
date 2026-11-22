
import numpy as np

def r_squared(y_true, y_pred):
    # Write your code here
    # r2 = 1 - (mse(y_true,y_pred)/mse(y_true,avg(y_true)))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    num = np.mean((y_true-y_pred)**2)
    y_mean = np.mean(y_true)
    den = np.mean((y_true-y_mean)**2)
    
    return 1-(num/den)