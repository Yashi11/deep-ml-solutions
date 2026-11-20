import numpy as np

def lr_range_finder(X, y, w0, a, b, n_steps):
    """
    Sweep learning rate from 10**a to 10**b over n_steps full-batch GD updates
    on a linear regression model (no bias). Return the list of MSE losses after
    each update.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w0 = np.asarray(w0, dtype=float)

    losses = np.zeros(n_steps)
    step = n_steps
    n = X.shape[0]

    for i in range(n_steps):
        step -=1
        if n_steps==1:
            lr=10**a
        else:
            lr = 10**(a+(b-a)*(i)/(n_steps-1))
        L = (2/n)*X.T@(X@w0-y)
        w0 = w0-lr*L
        loss= np.mean((X@w0-y)**2) #MSE is the mean of squares of error
        losses[i]=loss
    return losses

