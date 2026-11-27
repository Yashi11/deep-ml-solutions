import numpy as np
import math

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    # Your code here
    X_ = np.column_stack([np.ones(X.shape[0]),X])
    beta_hat = np.zeros(X_.shape[1]) # includes bias as well
    losses = []

    while iterations:
        z = X_@beta_hat
        y_pred = sigmoid(z)
        # deltaL/deltaw = (pi-yi)xi
        Lw = X_.T@(y_pred-y)

        loss = -np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
        losses.append(loss)

        beta_hat = beta_hat - learning_rate*Lw
        iterations -=1
    return np.round(beta_hat,4), np.round(losses,4)
    
