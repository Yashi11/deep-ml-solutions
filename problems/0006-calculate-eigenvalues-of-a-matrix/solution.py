def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    import numpy as np
    trace_a = matrix[0][0]+matrix[1][1]
    det_a = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    #Solving the eqn we get eigen_val = (trace +- sqrt(trace**2 - 4*det))/2
    discriminant = trace_a**2 - 4*det_a
    if discriminant<0:
        raise ValueError
    else:
        tmp = np.sqrt(discriminant)
        eigen1 = (trace_a + tmp)/2
        eigen2 = (trace_a - tmp)/2
    return [eigen1,eigen2]