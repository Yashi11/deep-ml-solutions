import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D numpy array of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    # Your code here
    data = np.asarray(data, dtype=float)
    n = len(data)
    mu = np.mean(data)
    var = (1/n)*np.sum((data-mu)**2)
    return mu,var