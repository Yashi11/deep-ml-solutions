import numpy as np

def compute_partial_derivative_poly2d(point: tuple[float, ...])-> tuple[float, ...]:
    x = point[0]
    y = point[1]
    return (2*x*y + y**2, x**2 + 2*x*y)

def compute_partial_derivative_exp_sum(point: tuple[float, ...])-> tuple[float, ...]:
    x = point[0]
    y = point[1]
    return (np.exp(x+y),np.exp(x+y))

def compute_partial_derivative_product_sin(point: tuple[float, ...])-> tuple[float, ...]:
    x = point[0]
    y = point[1]
    return (np.sin(y),x*np.cos(y))

def compute_partial_derivative_poly3d(point: tuple[float, ...])-> tuple[float, ...]:
    x = point[0]
    y = point[1]
    z = point[2]
    return (2*x*y, x**2 + z**2, 2*y*z)
def compute_partial_derivative_squared_error(point: tuple[float, ...])-> tuple[float, ...]:
    x = point[0]
    y = point[1]
    return (2*x - 2*y, 2*y-2*x)

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    if func_name is 'poly2d' and len(point)==2:
        ans = compute_partial_derivative_poly2d(point)
    elif func_name is 'exp_sum' and len(point)==2:
        ans = compute_partial_derivative_exp_sum(point)
    elif func_name is 'product_sin' and len(point)==2:
        ans = compute_partial_derivative_product_sin(point)
    elif func_name is 'poly3d' and len(point)==3:
        ans = compute_partial_derivative_poly3d(point)
    elif func_name is 'squared_error' and len(point)==2:
        ans = compute_partial_derivative_squared_error(point)
    else:
        raise ValueError
    return ans