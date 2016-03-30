import numpy as np

'''
INPUT:
X       = vector of data
OUTPUTS:
lower   = the lower bound for outliers
upper   = the upper bound for outliers
'''
def get_outlier_bounds(X):
    [lq, uq] = np.percentile(X, [25, 75])
    iqr = uq - lq
    lower = lq - (1.5 * iqr)
    upper = uq + (1.5 * iqr)
    return [lower, upper]
