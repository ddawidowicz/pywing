'''
This module
INPUTS:

OUTPUTS:

USAGE:
'''
import os
import sys
import pudb
import time
import timeit
import locale
import itertools
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy import linalg
from datetime import datetime
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.mixture import GMM, DPGMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from data_util import *
from plot_util import *
from create_metrics import *
from cluster_usage import *

locale.setlocale(locale.LC_ALL, 'en_US')
pd.options.display.width = 150 #set the display width for printing pandas df
np.set_printoptions(linewidth=150) #same for numpy


# =============================== GAP STATISTIC ============================== #
'''
This function is used for the gap statistic and finds the bounding box 
around all the data points (think smallest box that contains all points)
INPUTS:
X       = The original data matrix
OUPUTS:
xmin    = left-most x-value for bounding box
xmax    = right-most x-value for bounding box
ymin    = lower-most y-value for bounding box
ymax    = upper-most y-value for bounding box
'''
def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)


'''
This function is used for the gap statistic and finds the sum of the squared
distances from each data point to the mean of its assigned cluster for each
cluster and then finds the sum of these sums across clusters.
INPUTS:
mu          = A list of the cluster centroids
clusters    = A dictionary with the cluster label as the key and a np.array
              of data points as the value.
OUTPUT
A value for W_k, which is the sum across clusters of the sum of squared 
distances between each data point and the centroid for its assigned cluster.
'''
def Wk(mu, clusters):
    K = len(mu)
    return np.sum([np.linalg.norm(c-mu[i])**2/(2*len(c)) \
                   for i in range(K) for c in clusters[i]])


'''
This function is used for the gap statistic and creates a dictionary for each
cluster where the key is the cluster label and the value is a np.array of the
data points in that cluster.
INPUTS:
X           = The original data matrix
lbls        = A vector for each row of X giving the assigned cluster for that 
              point
OUTPUT:
clusters    = A dictionary with cluster label as the key and a np.array of
              data points as the value
'''
def get_cluster_dict(X, lbls):
    clusters = {}
    for lbl in np.unique(lbls):
        clusters[lbl] = X[lbls==lbl]
    return clusters


'''
This function computes the statistics necessary to a gap analysis of the 
number of clusters.
INPUTS:
X           = The data matrix
k_means_arr = An array with the kmeans models as elements
k_range     = The list of possible K's, e.g. 1-10
num_ref     = The number of reference samples to pull
OUTPUTS:
Wks         = An array of the W_k values for each value of K=k_i. The W_k
              value is a sum across clusters of the sum of squared distances
              between each data point and the centroid of its assigned cluster.
Wkbs        = An array of the Wks values for the reference distribution. 
              The reference distribution is a uniform sampling over the 
              bounding box.
sk          = An array of the sk values for each value of K=k_i. The sk value
              is \sqrt{1 + \frac{1}{B}} * std(Wkbs)
reference,
https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
'''
def get_gap_stats(X, k_means_arr, k_range, num_ref):
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    Wks = np.zeros(len(k_range))
    Wkbs = np.zeros(len(k_range))
    sk = np.zeros(len(k_range))
    for k_idx, k in enumerate(k_range):
        mu = k_means_arr[k_idx].cluster_centers_
        lbls = k_means_arr[k_idx].labels_
        clusters = get_cluster_dict(X, lbls)        
        Wks[k_idx] = np.log(Wk(mu, clusters))
        
        # Create num_ref reference datasets
        BWkbs = np.zeros(num_ref)
        for i in range(num_ref):
            Xb = []
            for n in range(X.shape[0]):
                Xb.append([np.random.uniform(xmin,xmax),
                          np.random.uniform(ymin,ymax)])
            Xb = np.array(Xb)
            k_means_b_arr = [KMeans(n_clusters=k).fit(Xb) for k in k_range]
            mu_b = k_means_b_arr[k_idx].cluster_centers_
            lbls_b = k_means_b_arr[k_idx].labels_
            clusters = get_cluster_dict(Xb, lbls_b)
            BWkbs[i] = np.log(Wk(mu_b, clusters))
        Wkbs[k_idx] = np.sum(BWkbs)/float(num_ref)
        #sk[k_idx] = np.sqrt(np.sum((BWkbs-Wkbs[k_idx])**2)/num_ref) #std(BWkbs)
        sk[k_idx] = np.std(BWkbs)
    sk = sk*np.sqrt(1+1/float(num_ref))
    return [Wks, Wkbs, sk]





# =================================== DPGMM ================================= #
def get_best_dpgmm(X, num_c, cv_type, alpha, iters, n_init, rand_state=None):
    best_bic = np.inf
    bic_dpgmm = None
    lbl_vec_dpgmm = np.zeros(X.shape[0])
    prob_vec_dpgmm = np.zeros(X.shape[0])
    log_prob_dpgmm = None
    for i in xrange(n_init):
        dpgmm = DPGMM(n_components=num_c, covariance_type=cv_type, \
                        alpha=alpha, random_state=rand_state)
        dpgmm.fit(X)
        b = dpgmm.bic(X)
        if b < best_bic:
            bic_dpgmm = b
            lbl_vec = dpgmm.predict(X)
            prob_vec = dpgmm.predict_proba(X)
            log_prob_dpgmm = np.sum(dpgmm.score(X))
    return [lbl_vec, prob_vec, bic_dpgmm, log_prob_dpgmm]


'''
This function gives the significance for each feature column between all 
clusters.
INPUTS:
df_clusters     = A list of data frames containing the columns of interest for
                  the resulting clusters.
                  df_clusters = [pd.DataFrame(X[lbl_vec==0]), \
                                 pd.DataFrame(X[lbl_vec==1]), \
                                 pd.DataFrame(X[lbl_vec==2])]
plot_it         = If true, plot a histogram of the significane of each column
verbose         = If true, print the significance for each column 
OUTPUT:
This function prints as it goes and shows the plot if plot_it==True
'''
def get_cluster_signif(df_clusters, plot_it=False, verbose=False):
    idx = range(len(df_clusters))
    combos = itertools.combinations(idx, 2)
    for combo in combos:
        a = combo[0]
        b = combo[1]
        df_a = df_clusters[a]
        df_b = df_clusters[b]
        print 'Cluster %d with cluster %d' % (a, b)
        signif = []
        for c in df_a.columns:
            arr_a = np.array(df_a[c])
            arr_b = np.array(df_b[c])
            arr_a = arr_a[~np.isnan(arr_a)]
            arr_b = arr_b[~np.isnan(arr_b)]
            p = stats.ttest_ind(arr_a, arr_b, equal_var=False)[1]
            if np.isnan(p):
                p = 0.0 #e.g. all day_deltas == 0
            signif.append(p)
            if verbose:
                print '%-15s\t%.4f' % (c, p)

        mean_signif = np.nanmean(signif)
        med_signif = np.percentile(signif, [50])[0] 
        print '-' * 60
        print '%15s\t%15s' % ('mean p-value', 'median p-value')
        print '%15.4f\t%15.4f' % (mean_signif, med_signif) 
        print ''
        print '=' * 60
        
        if plot_it:
            fig, ax = plt.subplots()
            ax.hist(signif, bins=np.arange(0, 1, 0.05))
            t1 = 'Significance of Clusters %d and %d' % (a, b)
            t2 = '\nMean=%.4f, Median=%.4f' % (mean_signif, med_signif)
            ax.set_title(t1+t2)
            ax.set_xlabel('p-value')
            ax.set_ylabel('Count of columns')
            plt.show()


'''
This function takes a training set and a testing set and classifies the data
in the testing set based on a classifier trained on the training set.
INPUTS:
X_trn   = The data array for training
y_trn   = The vector of cluster labels for X_trn
X_tst   = The data array for classifying
y_tst   = If available, this will be used to score the classifier
n_neigh = The K in the KNN classifier, the number of neighbors to use
OUTPUT:
y_pred  = The predicted cluster labels for X_tst
acc     = If y_tst is provided this is the accuracy of the classifier
'''
def classify_clusters(X_trn, y_trn, X_tst, y_tst=None, n_neigh=3):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(X_trn, y_trn) 
    y_pred = neigh.predict(X_tst)
    if y_tst is not None:
        acc = neigh.score(X_tst, y_tst)
    else:
        acc = None
    return [y_pred, acc]
       
       
# ==================================== EXTRA ================================= #
def get_cluster_df(df_sub, df_f, users, lbl_vec, cols):
    cluster1 = users[lbl_vec == 0]
    cluster2 = users[lbl_vec == 1]
    cluster3 = users[lbl_vec == 2]
    cluster4 = users[lbl_vec == 3]
    df1 = df_sub[df_sub['user'].isin(cluster1)][cols]. \
            merge(df_f[df_f['user'].isin(cluster1)] \
            [['user', 'survival_med']], how='outer', on='user')
    df2 = df_sub[df_sub['user'].isin(cluster2)][cols]. \
            merge(df_f[df_f['user'].isin(cluster2)] \
            [['user', 'survival_med']], how='outer', on='user')
    df3 = df_sub[df_sub['user'].isin(cluster3)][cols]. \
            merge(df_f[df_f['user'].isin(cluster3)] \
            [['user', 'survival_med']], how='outer', on='user')
    df4 = df_sub[df_sub['user'].isin(cluster4)][cols]. \
            merge(df_f[df_f['user'].isin(cluster4)] \
            [['user', 'survival_med']], how='outer', on='user')
    return [df1, df2, df3, df4]


def print_plot_df_cluster(df_c, cols, cnt):
    print 'shape df_%d = ' % cnt, df_c.shape
    print len(df_c['user'].unique()), 'users'
    for c in cols[1:]:
        if c == 'mean_revision_duration':
            print 'Median %s = %.3f min' % (c[5:], df_c[c].median()/60.0)
        elif c == 'survival_med':
            print 'Mean of %s = %.3f min' % (c, df_c[c].mean() / 60.0)
        elif c == 'event_hr':
            print 'Mode of %s = %.3f' % (c, df_c[c].mode())
        else:
            print 'Mean %s = %.3f' % (c, df_c[c].mean())
    df_c_log = df_c[cols[1:]].apply(func=np.log, axis=0)
    try:
        my_plt = plot_cluster_metrics(df_c_log)
    except:
        try:
            df_c_log = df_c_log.dropna(axis=1, how='all')
            my_plt = plot_cluster_metrics(df_c_log)
        except:
            print 'df_%d failed to plot' % cnt
            my_plt = None
    return my_plt


