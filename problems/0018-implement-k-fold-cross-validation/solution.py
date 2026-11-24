import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Your code here
    if shuffle:
        n_indexes = np.random.permutation(n_samples)
    else:
        n_indexes = np.arange(n_samples)
    folds = np.array_split(n_indexes,k)
    final_data = []
    for fold in folds:
        test = fold
        tmp = np.ones(n_samples,dtype=bool)
        tmp[fold]=False
        train = n_indexes[~np.isin(n_indexes,fold)]
        final_data.append((train,test))
    return final_data