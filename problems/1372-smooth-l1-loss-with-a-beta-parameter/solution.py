import numpy as np

def smooth_l1(pred, target, beta=1.0, reduction='mean'):
    """
    Elementwise Smooth L1 loss.

    Args:
        pred, target: arrays of the same shape
        beta: transition point between the quadratic and linear branches (> 0)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        float for 'mean'/'sum', array for 'none'
    """
    # Your code here
    if beta <=0:
        raise ValueError
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    x = pred-target

    mask = np.abs(x)<beta
    opt1 = (0.5*(x**2))/beta
    opt2 = np.abs(x)-0.5*beta
    loss_array = np.where(mask,opt1,opt2)

    if reduction is 'mean':
        return np.mean(loss_array)
    elif reduction is 'sum':
        return np.sum(loss_array)
    elif reduction is 'none':
        return loss_array
    else:
        raise ValueError
