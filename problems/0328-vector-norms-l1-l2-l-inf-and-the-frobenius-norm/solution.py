import numpy as np

def l1(arr: np.ndarray)->float:
    if arr.ndim==1:
        tmp=0.0
        for ele in arr:
            tmp = tmp+np.abs(ele)
        return tmp
    elif arr.ndim==2:
        tmp=0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                tmp= tmp+np.abs(arr[i][j])
        return tmp
    else:
        raise ValueError

def l2(arr: np.ndarray)->float:
    if arr.ndim==1:
        tmp=0.0
        for ele in arr:
            tmp = tmp+(ele**2)
        return np.sqrt(tmp)
    elif arr.ndim==2:
        tmp=0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                tmp= tmp+(arr[i][j]**2)
        return np.sqrt(tmp)
    else:
        raise ValueError

def linf(arr: np.ndarray)->float:
    if arr.ndim==1:
        tmp=0.0
        for ele in arr:
            tmp = max(np.abs(ele),tmp)
        return tmp
    elif arr.ndim==2:
        tmp=0.0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                tmp= max(np.abs(arr[i][j]),tmp)
        return tmp
    else:
        raise ValueError

def frobenius(arr: np.ndarray)->float:
    if arr.ndim!=2:
        raise ValueError
    else:
        return l2(arr)

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    if norm_type is 'l1':
        return l1(arr)
    elif norm_type is 'l2':
        return l2(arr)
    elif norm_type is 'linf':
        return linf(arr)
    elif norm_type is 'frobenius':
        return frobenius(arr)
    else:
        raise ValueError
    pass
