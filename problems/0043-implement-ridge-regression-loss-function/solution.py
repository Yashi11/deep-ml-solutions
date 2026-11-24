import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    X = np.asarray(X,dtype=float)
    w = np.asarray(w, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    rss = np.mean((X@w - y_true)**2)
    ridge_penalty = np.sum(w**2)*alpha
    return rss+ridge_penalty
