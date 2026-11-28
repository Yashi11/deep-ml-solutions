
from collections import Counter
import numpy as np

def confusion_matrix(data):
    # Implement the function here
    data = np.asarray(data,dtype=int)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for d in data:
        if d[0]==d[1]:
            if d[0]==1:
                tp+=1
            elif d[0]==0:
                tn+=1
        else:
            if d[0]==1:
                fn+=1
            elif d[0]==0:
                fp+=1
    return [[tp, fn], [fp, tn]]

