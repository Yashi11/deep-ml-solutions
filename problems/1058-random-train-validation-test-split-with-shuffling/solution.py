import numpy as np
import math

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    n = data.shape[0]
    sample = np.random.default_rng(seed).permutation(n)

    ## adding them after multiplying by n - as floating point rounding can result in error - lower index
    n_train = math.floor(train_frac * n)
    n_val = math.floor(validation_frac * n)

    train_end = n_train
    val_end = train_end + n_val

    train_sample = sample[:train_end]
    val_sample = sample[train_end:val_end]
    test_sample = sample[val_end:]

    return [data[train_sample],data[val_sample],data[test_sample]]