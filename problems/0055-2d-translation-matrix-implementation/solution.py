import numpy as np
def translate_object(points, tx, ty):
    points = np.asarray(points, dtype=float)
    t = np.array([tx,ty]).astype(float)
    tmp = np.ones_like(points)
    translated_points = points + t*tmp
    return translated_points
