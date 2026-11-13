import numpy as np

def woodbury_solve(sigma_diag, X, B):
    sigma_diag = np.asarray(sigma_diag) #[diag1,diag2...]
    sigma_diag_inv = 1/sigma_diag # By doing element wise division [1/diag1,1/diag2...]

    X = np.asarray(X)
    B = np.asarray(B)
    X_t = X.T
    d = X.shape[1]
    I = np.identity(d)
    sigma_inv = sigma_diag_inv.reshape(-1,1)
    
    original_shape = B.shape
    if B.ndim ==1:
        B = B.reshape(-1,1)
    e_inv_x = sigma_inv * X
    e_inv_b = sigma_inv * B #incorporate B in the product 
    tmp = np.linalg.solve(I + X_t@e_inv_x, X_t@e_inv_b)
    ans = e_inv_b - e_inv_x@tmp #incorporate B in the product make e_inv to e_inv_b
    return ans.reshape(original_shape)
    

def sherman_morrison(A_inv, u, v):
    v = np.asarray(v).reshape(-1,1)
    u = np.asarray(u).reshape(-1,1)
    v_t = v.T
    val = 1 + v_t@A_inv@u
    if val==0:
        raise ValueError
    else:
        ans = A_inv - ((A_inv@u@v_t@A_inv)/(val))
        return ans

def logdet_rank_one_update(logdet_A, A_inv, u, v):
    v = np.asarray(v).reshape(-1,1)
    u = np.asarray(u).reshape(-1,1)
    v_t =v.T
    val = (1 + v_t@A_inv@u).item() # make scalar
    if val <= 0:
        raise ValueError
    else:
        return float(logdet_A + np.log(val))
