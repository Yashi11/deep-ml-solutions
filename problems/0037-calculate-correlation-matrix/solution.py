import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # cov(X,Y) = (1/n)*sigma((x-E(x))*(y-E(y)))
    ## corr(X,Y) = cov(X,Y)/np.sqrt(var(X)*var(Y))
    if Y is None:
        Y = X
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")

    varx = np.var(X, axis=0)
    vary = np.var(Y, axis=0)
    ex = X.mean(axis=0)
    ey = Y.mean(axis=0)
    n = X.shape[0]
    X_centred = X-ex
    Y_centred = Y-ey
    cov_X_Y = (1/n)*(X_centred.T@Y_centred)
    corr_X_Y = cov_X_Y/np.sqrt(varx[:,None]*vary[None,:])
    return corr_X_Y