# Usage:
# normalize(X)
#
# Input:
# X =   a numpy array of numbers where each column is a different feature
#       and each row is a different observation
#
# Outputs:
# X_norm = normed X array (mean=0, std=1)
# mu = mean used in norming
# sigma = std used in norming
#

import numpy as np

def normalize(X):
    num_r = X.shape[0]
    num_c = X.shape[1]
    mu = np.mean(X,axis=0) #by column
    sigma = np.std(X,axis=0, ddof=1) #by column and use sample deviation (N-1)
    X_norm = (X-mu)/sigma
    return (X_norm, mu, sigma)
    
def tester():
#expected result
#-1, -1
# 0,  0 
# 1,  1
    X = np.array([[1,2],[3,4], [5,6]])
    (X_norm, mu, sigma) = normalize(X)

tester()
