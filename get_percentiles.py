import numpy as np


'''
Find the first quartile from an array
INPUT:
x   = An array-like object
OUPUT:
returns the first quartile of x
'''
def get_q1(x):
    try:
        return np.percentile(x, [25])[0]
    except:
        return np.percentile(x[~np.isnan(x)], [25])[0]


'''
Find the median from an array
INPUT:
x   = An array-like object
OUPUT:
returns the median of x
'''
def get_median(x):
    try:
        return np.percentile(x, [50])[0]
    except:
        return np.percentile(x[~np.isnan(x)], [50])[0]


'''
Find the median from an array
INPUT:
x   = An array-like object
OUPUT:
returns the median of x
'''
def get_q2(x):
    try:
        return np.percentile(x, [50])[0]
    except:
        return np.percentile(x[~np.isnan(x)], [50])[0]


'''
Find the third quartile from an array
INPUT:
x   = An array-like object
OUPUT:
returns the third quartile of x
'''
def get_q3(x):
    try:
        return np.percentile(x, [75])[0]
    except:
        return np.percentile(x[~np.isnan(x)], [75])[0]


