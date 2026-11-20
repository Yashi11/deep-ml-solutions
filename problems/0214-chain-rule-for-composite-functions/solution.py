import numpy as np

def derivative_square(x):
    return 2*x
def derivative_sin(x):
    return np.cos(x)
def derivative_exp(x):
    return np.exp(x)
def derivative_log(x):
    return 1/x

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    ans = 1.0
    values_of_x = []
    values_of_x.append(x)

    for function in functions[::-1]:
        if function is 'sin':
            x = np.sin(x)
            values_of_x.append(x)
        elif function is 'square':
            x = x**2
            values_of_x.append(x)    
        elif function is 'exp':
            x = np.exp(x)
            values_of_x.append(x)
        elif function is 'log':
            x = np.log(x)
            values_of_x.append(x)
        else:
            raise ValueError
    values_of_x = values_of_x[::-1]
    n = len(functions)
    for i in range(len(functions)):
        if functions[i] is 'sin':
            derivative = derivative_sin(values_of_x[i+1])
        elif functions[i] is 'square':
            derivative = derivative_square(values_of_x[i+1])
        elif functions[i] is 'exp':
            derivative = derivative_exp(values_of_x[i+1])
        elif functions[i] is 'log':
            derivative = derivative_log(values_of_x[i+1])
        else:
            raise ValueError
        ans = ans*derivative
    return np.round(ans,4)
