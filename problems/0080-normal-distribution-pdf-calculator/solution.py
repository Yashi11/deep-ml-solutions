import math
import numpy as np
def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    # Your code here
    mean = float(mean)
    std_dev = float(std_dev)
    x = float(x)
    z = (-1/2)*(((x-mean)/std_dev)**2)
    pdf = (1/(std_dev*np.sqrt(2*np.pi)))*(np.e**(z))  
    return round(pdf,5)