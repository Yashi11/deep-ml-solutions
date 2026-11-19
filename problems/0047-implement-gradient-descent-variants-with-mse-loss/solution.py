import numpy as np

def get_L(X, y, weights):
    # L = X.T@(X@w - y)/m
    m = X.shape[0]
    prediction = X@weights
    error = prediction-y
    L = (X.T@error)*(2/m)
    return L

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float).copy()

    while n_epochs:
        if method is 'batch':
            L = get_L(X,y,weights)
            weights = weights - learning_rate*L
        elif method is 'stochastic':
            for i in range(X.shape[0]):
                L = get_L(X[i:i+1],y[i:i+1],weights)
                weights = weights - learning_rate*L
        elif method is 'mini_batch':
            for i in range(0,X.shape[0],batch_size):
                L = get_L(X[i:i+batch_size],y[i:i+batch_size],weights)
                weights = weights - learning_rate*L
        else:
            raise ValueError
        n_epochs = n_epochs -1
    return weights

