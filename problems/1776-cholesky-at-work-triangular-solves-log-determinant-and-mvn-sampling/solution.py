import numpy as np

def forward_sub(L, b):
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    y = np.zeros(n,dtype=float)
    for i in range(n):
        sum = 0.0
        j = 0
        while j<i:
            sum = sum + (L[i][j]*y[j])
            j = j+1
        if L[i][i]==0:
            raise ValueError
        y[i] = (b[i]-sum)/L[i][i]
    return y
def back_sub(U, y):
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)
    n = U.shape[0]
    x = np.zeros(n,dtype=float)
    for i in range(n-1,-1,-1):
        sum = 0.0
        j = n-1
        while j>i:
            sum = sum + (U[i][j]*x[j])
            j = j-1
        if U[i][i]==0:
            raise ValueError
        x[i] = (y[i]-sum)/U[i][i]
    return x

def chol_solve(A, b):
    #Ax=b means LL.Tx=b meaning L(L.Tx)=b 
    ## considering y=L.Tx y = forward_sub(L,b)
    ## now L.Tx=y so x = back_sub(L.T,y)
    L = np.linalg.cholesky(A)
    y = forward_sub(L,b)
    x = back_sub(L.T,y)
    return x

def logdet_chol(A):
    L = np.linalg.cholesky(A)
    sum = 0
    n = L.shape[0]
    for i in range(n):
        sum = sum + np.log(L[i][i])
    return 2*sum


def mvn_sample(mu, Sigma, eps):
    L = np.linalg.cholesky(Sigma)
    ans = []
    eps = np.asarray(eps, dtype=float)
    m = eps.shape[0]
    for i in range(m):
        ansi = L@eps[i]+mu
        ans.append(ansi)
    return np.asarray(ans)
