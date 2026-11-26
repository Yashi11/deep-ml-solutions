import numpy as np


def model_fit_stats(X: np.ndarray, y: np.ndarray) -> tuple:
    """Residual standard error, R-squared and the overall F-statistic.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (rse, r2, f_stat) as floats. RSS == 0 gives (0.0, 1.0, inf).
    """
    # Your code here
    X = np.asarray(X, dtype=float)
    y= np.asarray(y, dtype=float)
    n =X.shape[0]
    p = X.shape[1]
    X_with_intercept = np.column_stack((np.ones(n), X))
    theta,_,_,_ = np.linalg.lstsq(X_with_intercept,y, rcond=None)
    y_pred = X_with_intercept@theta
    rss = float(np.sum((y-y_pred)**2))
    tss = float(np.sum((y-y.mean())**2))
    tol = np.finfo(float).eps**2 * max(1.0, float(np.sum(y**2))) * n
    if rss<=tol:
        return (0.0, 1.0, float('inf'))
    r2 = 1-(rss/tss)
    rse = np.sqrt(rss/(n-p-1))
    f_stat = ((tss-rss)/p)/(rss/(n-p-1))
    return (rse,r2,f_stat)
