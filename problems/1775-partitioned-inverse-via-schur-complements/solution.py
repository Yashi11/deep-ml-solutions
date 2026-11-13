import numpy as np

def get_e(M,k):
    matrix_e = []
    for row in range(k):
        row_e = []
        for col in range(k):
            row_e.append(M[row][col])
        matrix_e.append(row_e)
    return matrix_e

def get_f(M,k):
    matrix_f = []
    n = len(M[0])
    for row in range(k):
        row_f = []
        for col in range(k,n):
            row_f.append(M[row][col])
        matrix_f.append(row_f)
    return matrix_f

def get_g(M,k):
    matrix_g = []
    m = len(M)
    for row in range(k,m):
        row_g = []
        for col in range(k):
            row_g.append(M[row][col])
        matrix_g.append(row_g)
    return matrix_g

def get_h(M,k):
    matrix_h = []
    n = len(M[0])
    m = len(M)
    for row in range(k,m):
        row_h = []
        for col in range(k,n):
            row_h.append(M[row][col])
        matrix_h.append(row_h)
    return matrix_h

def partitioned_inverse(M, k):
    e = np.array(get_e(M,k))
    f = np.array(get_f(M,k))
    g = np.array(get_g(M,k))
    h = np.array(get_h(M,k))

    h_inv = np.linalg.inv(h)

    s_h = e - f@(h_inv)@g
    s_h_inv = np.linalg.inv(s_h)

    M1 = s_h_inv
    M2 = -(s_h_inv@f@h_inv)
    M3 = -(h_inv@g@s_h_inv)
    M4 = h_inv + (h_inv@g@s_h_inv@f@h_inv)
    return np.block([[M1, M2],[M3,M4]])

def block_slogdet(M, k):
    e = np.array(get_e(M,k))
    f = np.array(get_f(M,k))
    g = np.array(get_g(M,k))
    h = np.array(get_h(M,k))

    h_inv = np.linalg.inv(h)
    s_h = e - f@(h_inv)@g
    det_s_h = np.linalg.det(s_h)
    sign_s_h,log_s_h = np.linalg.slogdet(s_h)
    sign_h, log_h = np.linalg.slogdet(h) 
    return sign_s_h*sign_h, log_h+log_s_h

def bordered_inverse(A_inv, b, c):
    b = np.asarray(b).reshape(-1,1)
    b_t = b.T
    s = c - b_t@A_inv@b
    if np.abs(s)<10**(-12):
        raise ValueError
    else:
        s_inv = 1/s
        M1 = A_inv + A_inv@b@s_inv@b_t@A_inv
        M2 = -(A_inv@b@s_inv)
        M3 = - (s_inv@b_t@A_inv)
        M4 = s_inv
        return np.block([[M1,M2],[M3,M4]])
