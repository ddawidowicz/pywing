import numpy as np

'''
This function finds the Shannon Entropy of an array X. This is used as a metric
for the diversity of events and apps.
INPUT:
X   = An array of the counts of apps or events, e.g. ([120, 23, 0, 12])
OUTPUT:
H   = The entropy metric
'''
def get_entropy(X):
    try:
        #if X is a matrix - compute per row
        px = X.astype('float') / np.sum(X, axis=1).reshape(-1, 1)
        log_px = np.where(px>0, np.log(px), 0)
        H = -np.sum((px * log_px), axis=1) + 0.0 #add 0.0 to remove neg from 0.0
    except:
        #if X is a vector - return single value
        px = X.astype('float') / X.sum()
        log_px = np.where(px>0, np.log(px), 0.0) 
        H = -(px * log_px).sum() + 0.0 #adding 0.0 removes neg from 0.0
    return H
