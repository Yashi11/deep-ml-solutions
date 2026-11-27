import math
import numpy as np
def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.asarray(scores, dtype=float)
    max_scores = max(scores)
    scores_ = scores - max_scores
    e_scores = np.array([math.e**val for val in scores_])
    sum_e_scores = np.sum(e_scores)
    return e_scores/sum_e_scores
