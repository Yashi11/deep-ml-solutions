import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    # Your implementation here
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(runs, n))
        mu, sigma = 0.5, np.sqrt(1 / 12)
    elif distribution == 'exponential':
        samples = np.random.exponential(1.0, size=(runs, n))
        mu, sigma = 1.0, 1.0
    elif distribution == 'bernoulli':
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        mu, sigma = 0.3, np.sqrt(0.3*0.7)
    else:
        raise ValueError
    sample_mean = np.mean(samples, axis=1)
    sample_mean_standardized = (sample_mean - mu)/(sigma/np.sqrt(n))
    return {'mean': float(np.mean(sample_mean_standardized)), 'std': float(np.std(sample_mean_standardized))}
