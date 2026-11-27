import numpy as np
import math

def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here
    x = np.asarray(x,dtype=float)
    x_ = x-np.max(x)
    e_x = np.array([math.e**val for val in x_])
    softmax_x = e_x/np.sum(e_x)

    J = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i==j:
                J[i][i]= softmax_x[i]*(1-softmax_x[i])
            else:
                J[i][j]= (-1)*softmax_x[i]*softmax_x[j]
    return J

