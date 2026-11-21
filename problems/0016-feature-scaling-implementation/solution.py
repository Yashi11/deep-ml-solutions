import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    mean = np.mean(data, axis=0)
    std = np.std(data,axis=0)
    standardized_data = (data-mean)/std

    min_ = np.min(data,axis=0)
    max_ = np.max(data,axis=0)
    normalized_data = (data-min_)/(max_-min_)
    
    return standardized_data, normalized_data