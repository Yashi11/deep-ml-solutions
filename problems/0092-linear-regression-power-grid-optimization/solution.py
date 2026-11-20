import math
import numpy as np
PI = 3.14159

def power_grid_forecast(consumption_data):
    # 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    # 2) Perform linear regression on the detrended data.
    # 3) Predict day 15's base consumption.
    # 4) Add the day 15 fluctuation back.
    # 5) Round, then add a 5% safety margin (rounded up).
    # 6) Return the final integer.
    consumption_data = np.asarray(consumption_data, dtype=float)
    n = len(consumption_data)
    days = np.arange(1,n+1,1)

    base_consumption_data = consumption_data - (10*np.sin((2*PI*(days))/10))
    
    X = np.column_stack((days, np.ones(n)))
    weights, _, _, _ = np.linalg.lstsq(X, base_consumption_data, rcond=None)
    
    prediction_day_15_base = weights[0]*15+weights[1]
    prediction_day_15 = prediction_day_15_base + 10*np.sin(((2*PI*15)/10))
    return math.ceil(np.round(prediction_day_15)*1.05)