import numpy as np
def rotation_layer(X, angle):
    X = np.asarray(X,dtype=float)
    rotation_matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    for i in range(len(X)):
        X[i]=rotation_matrix@X[i]
    return X.tolist()