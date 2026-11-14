import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    """
    if degree<0:
        return []
    ans = []
    for ele in data:
        x = ele
        tmp = 1
        ans_i = []
        for j in range(degree+1):
            ans_i.append(tmp)
            tmp = tmp*x
        ans.append(ans_i)
    return ans