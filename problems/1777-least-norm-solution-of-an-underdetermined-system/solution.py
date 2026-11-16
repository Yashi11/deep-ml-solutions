import numpy as np

def min_norm_solution(A, b):
    right = np.linalg.solve(A@A.T,b)
    return (A.T)@right

def min_norm_via_kkt(A, b):
    # L = x.x.T + lam.T(Ax-b)
    # d(L)/dx = 2x + lamA = 0
    # d(L)/dlam = Ax-b =0
    # so x = solve(A,b) and lam=solve(A,-2x)
    # We will express the 2 equations as block 
    # [[2I, A.T],[A,0]]@[x,lam]=[0,b]
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    m, n = A.shape

    # Build the (n + m) × (n + m) KKT system
    kkt = np.block([
        [2 * np.eye(n), A.T], # np.eye creates any matrix (even non square) with 1 on diagonals and 0 everywhere else
        [A, np.zeros((m, m))]
    ])

    rhs = np.concatenate([np.zeros(n), b])

    solution = np.linalg.solve(kkt, rhs)

    x = solution[:n]
    lam = solution[n:]

    return x, lam



def min_norm_from_particular(A, x_p):
    b = A@x_p
    return min_norm_solution(A,b)
