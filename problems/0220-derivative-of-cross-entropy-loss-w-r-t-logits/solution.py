import math
import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)
        
    Returns:
        Gradient vector where gradient[i] = dL/d(logits[i])
    """
    # Your code here
    y = np.zeros_like(logits)
    y[target]=1
    logits = np.asarray(logits,dtype=float)
    logits_ = logits - np.max(logits)
    e_logits = np.array([math.e**val for val in logits])
    q_logits = e_logits/np.sum(e_logits)
    result = q_logits - y
    return result