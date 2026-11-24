import numpy as np

def soft_threshold(w: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft-thresholding operator element-wise.
    
    S(w, λ) = sign(w) * max(|w| - λ, 0)
    
    Args:
        w: Input array
        threshold: Threshold value λ
    
    Returns:
        Soft-thresholded array where:
        - Values with |w| > λ are shrunk toward zero by λ
        - Values with |w| ≤ λ become exactly zero
    """
    # Your code here
    ans = np.sign(w)*np.maximum(np.abs(w)-threshold,0)
    return ans

def l1_regularization_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    """
    Implement Lasso Regression using ISTA (Iterative Shrinkage-Thresholding Algorithm).
    
    ISTA alternates between:
    1. Gradient step on MSE loss: w_temp = w - lr * gradient_mse
    2. Proximal step (soft-thresholding): w_new = soft_threshold(w_temp, lr * alpha)
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        alpha: L1 regularization strength
        learning_rate: Step size for gradient descent
        max_iter: Maximum iterations
        tol: Convergence tolerance on weight change
    
    Returns:
        tuple: (weights, bias)
    
    Note: The bias term is NOT regularized.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y,dtype=float).reshape(-1)
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    
    # Your code here
    while max_iter>0:
        Lw = (1/n_samples)*(X.T@(X@weights + bias - y))
        Lb = np.mean(X@weights + bias - y)
        weights_tmp = weights - learning_rate*Lw
        weights_new = soft_threshold(weights_tmp,learning_rate*alpha)
        bias_new = bias - learning_rate*Lb
        weight_change = np.linalg.norm(weights_new-weights)
        weights = weights_new
        bias = bias_new
        if np.linalg.norm(weight_change)<tol:
            break        
        max_iter-=1
    return weights,bias

