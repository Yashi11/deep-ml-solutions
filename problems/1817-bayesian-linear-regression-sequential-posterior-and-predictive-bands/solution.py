import numpy as np

def posterior(X, y, m0, s0, sigma2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m0 = np.asarray(m0, dtype=float)
    s0 = np.asarray(s0, dtype=float)

    s0_inv = np.linalg.inv(s0)

    sn = np.linalg.inv(s0_inv + (1/sigma2)*X.T@X)
    mn = sn@(s0_inv@m0 + (1/sigma2)*X.T@y)
    return mn,sn

def sequential_posterior(X, y, m0, s0, sigma2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mn = np.asarray(m0, dtype=float)
    sn = np.asarray(s0, dtype=float)
    means = []

    for i in range(X.shape[0]):
        mn, sn = posterior(X[i:i+1],y[i:i+1],mn,sn,sigma2)
        means.append(mn)

    return mn,sn,np.asarray(means)

def predictive(X_test, mN, sN, sigma2):
    # return (mean, var)
    X_test = np.asarray(X_test, dtype=float)
    mn = np.asarray(mN, dtype=float)
    sn = np.asarray(sN, dtype=float)

    means = []
    variances = []
    
    for i in range(X_test.shape[0]):
        mean = mn.T@X_test[i]
        variance = sigma2 + X_test[i].T@sn@X_test[i]
        means.append(mean)
        variances.append(variance)
    return np.asarray(means),np.asarray(variances)

