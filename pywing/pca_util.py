'''
This module contains functions for handling PCA
'''

import pudb
import time
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
This function plots the PCA error as a function of the number of principle
components used.
Input:
X   = a numpy array, num_observations x num_features
Output:
graph is displayed, but not saved
'''
def plot_pca_err(X):
    [U, S, V] = linalg.svd(X, full_matrices=False, compute_uv=True,
                            overwrite_a=True)
    err = []
    for i in range(len(S)):
        ratio = np.sum(S[:i])/np.sum(S)
        err.append(1-ratio)

    err = np.array(err)
    plt.plot(range(1,len(S)), err[1:]) #start at 1 b/c err[0] is always 1
    plt.show()

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(U[:,0], U[:,1], U[:,2])


'''
Plot the explained variance against the number of pca components. Note that the 
x-values show 0-indexed numbers so the actual number of components is 1 more
than the n that is shown.
INPUT:
ex_var  = pca.explained_variance_ratio_ array where pca is the fitted pca 
          model from sci-kit learn 
          >> from sklearn.decomposition import PCA
          >> pca = PCA().fit(X) 
          >> pca.explained_variance_ratio_
best_c  = If this is not None a red star will be placed in this position (0-idx)
max_x   = The max x (number of components) to display
fig_sz  = A tuple with the (height, width) of figure
OUTPUT:
fig     = The figure object
'''
def plot_pca_variance(ex_var, best_c=None, max_x=None, fig_sz=None):
    #y = ex_var/ex_var.sum() #if use explained_variance_ not _ratio_
    x = np.arange(ex_var.shape[0])
    
    if fig_sz:
        fig, ax = plt.subplots(figsize=fig_sz)
    else:
        fig, ax = plt.subplots()
    ax.plot(x, ex_var, 'b-', lw=3)
    if best_c:
        tot_var = np.cumsum(ex_var[:best_c+1]/ex_var.sum())[-1]
        txt = '(n=%d, variance=%.1f%%)' % (best_c, (100.0 * tot_var))        
        title = 'Explained variance vs. the number of PCA components\n' + \
                '(%.1f%% of variance explained)' % (100.0 * tot_var)

        ax.plot(best_c, ex_var[best_c]/ex_var.sum(), 'r*', ms=20)
        ax.text( best_c+2, ex_var[best_c]/ex_var.sum()*1.2, txt, fontsize=14)
        ax.set_title(title, fontsize=18)
    else:
        ax.set_title('Explained variance vs. the number of PCA components')
    if max_x:
        ax.set_xlim([0, max_x])
    #new_lbls = [str(item+1) for item in plt.xticks()[0]] #0-idx
    #ax.set_xticklabels(new_lbls) #new lbls 1-idx
    ax.set_xlabel('Number of PCA components', fontsize=14)
    ax.set_ylabel('Explained variance', fontsize=14)
    return fig


'''
This function returns a PCA version of the data. Remember to use this on the
training and testing X data.
Inputs:
X           = numpy array, num_observations x num_features, with the X data
num_cols    = number of principle components to select (use the function
              above to diagnose this)
cols        = an optional list of specific columns to use (use this if the
              data lacks lots of collinearity, but has high correlation
              between certain columns of X and the target value.
Output:
Z           = The PCA version of the X data
'''
def get_pca_data(X, num_cols, cols=[]):
    cov = 1/X.shape[0] * np.dot( np.transpose(X), X)
    [U,S,V] = linalg.svd(cov, full_matrices=True, compute_uv=True)

    if len(cols) > 0: #select specific columns to use
        U_reduce = U[:,cols] 
    else:
        U_reduce = U[:, range(num_cols)]
    Z = np.dot( X, U_reduce)
    return Z



if __name__ == '__main__':
    start = time.time()
    data = np.genfromtxt('data/dc1_trn_num5k.csv', delimiter=',')
    X = data[:,0:15]
    y = data[:,[15]]

    plot_pca_errs(X) 
    end = time.time()
    print 'Elapsed time = %.4f seconds' % (end-start)
