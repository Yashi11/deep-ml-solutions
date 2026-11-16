import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if x0 is None:
        x0 = np.zeros_like(b)
    else:
        x0 = np.asarray(x0, dtype=float)
    r0 = b-A@x0
    p0 = r0.copy() # must be a copy not p0=r0

    if np.linalg.norm(r0) < tol: # if current residual is < tolerance return x0
        return x0

    for i in range(n):
        alpha0 = (r0@r0)/(p0@A@p0) # step size
        x1 = x0 + alpha0*p0 #updated soln
        r1 = r0 - alpha0*A@p0 # updated residual
        if np.linalg.norm(r1) < tol:
            return x1
        b0 = (r1@r1)/(r0@r0) #direction scaling based on residual now vs before
        p1 = r1 + b0*p0 # new direction
        x0 = x1 # update new as old for next iteration
        r0 = r1
        p0=p1
    return x0
