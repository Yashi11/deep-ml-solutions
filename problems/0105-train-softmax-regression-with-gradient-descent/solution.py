import numpy as np
def softmax(z):
    z_ = z - np.max(z,axis=1, keepdims=True)
    e_z = np.exp(z_)
    return e_z/np.sum(e_z,axis=1, keepdims=True)

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
    """
    # Your code here
    n,p = X.shape
    X_ = np.column_stack([np.ones(n),X])
    
    # we need to get all the classes
    n_classes = int(np.max(y)) + 1
    weights = np.zeros((n_classes, p+1))
    
    losses = []
    while iterations:
        q = softmax(X_@weights.T)
        correct_probs = q[np.arange(n), y]
        loss = -np.sum(np.log(correct_probs))
        losses.append(loss)

        y_hot_one = np.eye(n_classes)[y]
        gradient = (q-y_hot_one).T@X_
        weights = weights - learning_rate*gradient

        iterations -=1

    return np.round(weights,4), np.round(losses,4)