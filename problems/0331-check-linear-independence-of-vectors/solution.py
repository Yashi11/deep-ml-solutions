import numpy as np

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your code here
    if len(vectors)==0:
        return True
    vectors = np.asarray(vectors,dtype=float)
    if vectors.shape[1]==0:
        return False
    rank = np.linalg.matrix_rank(vectors)
    if rank < vectors.shape[0]: #rank should be equal to #of columns for independence
        return False
    else:
        return True