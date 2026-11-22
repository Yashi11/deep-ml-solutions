import numpy as np
from scipy import stats
def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here
    impute = 0
    if strategy is 'mean':
        impute = np.nanmean(data,axis=0)
    elif strategy is 'median':
        impute = np.nanmedian(data,axis=0)
    elif strategy is 'mode':
        impute = stats.mode(data,axis=0,nan_policy="omit",keepdims=False).mode
    data = np.where(np.isnan(data),impute,data)
    return data  
    