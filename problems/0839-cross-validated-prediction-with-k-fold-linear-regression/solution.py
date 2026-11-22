import numpy as np

def cross_val_predict(X, y, k):
    """
    Return out-of-fold predictions for every sample using k-fold
    cross-validation with a linear regression model (with intercept).
    """
    # Your code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]

    if k<2 or k>n:
        raise ValueError
    X_with_bias = np.column_stack((np.ones(n), X))

    indices = np.arange(n)
    folds = np.array_split(indices, k)

    predictions = np.empty(n,dtype=float)
    
    for val_indices in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_indices]=False
        X_train = X_with_bias[train_mask]
        y_train = y[train_mask]
        X_val = X_with_bias[val_indices]

        theta,_,_,_ = np.linalg.lstsq(X_train,y_train,rcond=None)

        predictions[val_indices] = X_val@theta

    return predictions



