import numpy as np

def online_simple_regression(xs, ys):
    # return (N, 2) array of (w0, w1) after each point
    xs = np.asarray(xs,dtype=float)
    ys=np.asarray(ys,dtype=float)
    weights = []
    running_xavg=0.0
    running_yavg=0.0
    cxx=0.0
    cxy=0.0
    seen = 0
    for i in range(len(xs)):
        dx = xs[i] - running_xavg
        dy = ys[i] - running_yavg

        seen = seen+1

        running_xavg += dx/seen
        running_yavg += dy/seen
        
        cxx += dx*(xs[i]-running_xavg)
        cxy += dx*(ys[i]-running_yavg)

        if cxx==0:
            w1 = 0.0
        else:
            w1=cxy/cxx
        w0 = running_yavg-w1*running_xavg
        weights.append([w0,w1])
    return np.asarray(weights, dtype=float)

