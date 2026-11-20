import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Using taylor's series
    # df(x)/dx = (f(x+epsilon)-2f(x)+f(x-epsilon))/(epsilon**2)
    x = np.asarray(x,dtype=float)
    analytical_grad = np.asarray(analytical_grad,dtype=float)
    numerical_grad = np.zeros_like(x)

    for i in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] = x_plus[i]+epsilon
        x_minus[i] = x_minus[i]-epsilon

        numerator = f(x_plus)-f(x_minus)
        denominator = 2*epsilon
        numerical_grad[i] = numerator/denominator

    denominator_diff = (np.linalg.norm(numerical_grad)+np.linalg.norm(analytical_grad))
    if denominator_diff==0: return(numerical_grad,0.0)
    diff = np.linalg.norm(numerical_grad-analytical_grad)/denominator_diff
    return (numerical_grad,float(diff))
    