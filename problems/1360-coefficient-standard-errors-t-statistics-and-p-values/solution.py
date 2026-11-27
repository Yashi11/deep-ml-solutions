import math

import numpy as np

def phi(z: np.ndarray) -> np.ndarray:
    return np.array([0.5*(1+math.erf(value/np.sqrt(2))) for value in z])

def ols_inference(X: np.ndarray, y: np.ndarray) -> tuple:
    """Least-squares fit with an intercept, plus inference on each coefficient.

    Args:
        X (np.ndarray): (n, p) design matrix without an intercept column.
        y (np.ndarray): (n,) target.

    Returns:
        tuple: (coef, se, t, p), each a length-(p+1) array, intercept first.
    """
    # Your code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y,dtype=float)
    n,p = X.shape

    D = np.column_stack([np.ones(n),X])
    
    # lstsq gives the entire coeff beta = np.inv(D.T@D)@D.T@y
    beta_hat = np.linalg.lstsq(D,y,rcond=None)[0]
    y_pred = D@beta_hat
    rss = np.sum((y-y_pred)**2)
    sigma_hat = np.sqrt(rss/(n-p-1))
    # This is the covariance matrix Cov(beta_hat)=sigma2 * np.linalg.inv(D.T@D)
    var_b_hat = (sigma_hat**2)*(np.linalg.inv(D.T@D))
    se = np.sqrt(np.diag(var_b_hat))
    t = beta_hat/se
    p = 2*(1-phi(np.abs(t)))
    return beta_hat,se,t,p

