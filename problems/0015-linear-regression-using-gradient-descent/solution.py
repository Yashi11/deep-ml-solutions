import numpy as np

def get_L(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    m, n = X.shape
    x_pred = X@theta
    error = x_pred - y
    return (X.T@error)/m
    
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros
    while iterations:
        L = get_L(X,y,theta)
        theta_new = theta - alpha*L
        theta = theta_new
        iterations = iterations -1

    return theta.flatten()