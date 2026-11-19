import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(y, dtype=float)
    # w* = (XtX)_inv*(Xty)
    X_t_X =X.T@X
    X_t_Y = X.T@Y
    w = np.linalg.solve(X_t_X,X_t_Y)
    return np.round(w,4)