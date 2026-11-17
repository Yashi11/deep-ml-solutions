import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h

def create_hv(dim, seed):
    '''Creates a bipolar hypervector of given dimension using the seed.'''
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)

def create_row_hv(row, dim, random_seeds):
    '''Create composite hypervector for a dataset row.
    
    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    '''
    bundle = np.zeros(dim, dtype=float)
    for ele in row.keys():
        name_vector = create_hv(dim,random_seeds[ele])
        value_seed = (random_seeds[ele]+deterministic_hash(row[ele]))%(2**32-1)
        value_vector = create_hv(dim, value_seed)
        bind = name_vector*value_vector
        bundle = bundle + bind # sum all bound vectors elelment wise
    mask = bundle[:]>=0
    return np.where(mask,1,-1) # map each entry to 1 if it is ≥ 0 and to −1 otherwise