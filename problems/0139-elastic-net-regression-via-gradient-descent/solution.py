import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    while max_iter>0:
        e = X@w + b - y
        Lw = (1/n_samples)*X.T@e + alpha1*np.sign(w) + 2*alpha2*w
        Lb = (1/n_samples)*np.sum(e)
        w = w - learning_rate*Lw
        b = b - learning_rate*Lb
        if np.linalg.norm(Lw)<tol:
            break
        max_iter-=1
    return w,b