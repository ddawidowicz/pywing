import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
sns.set(style="darkgrid")


'''
This function plots the elbow curve for kmeans
INPUTS:
X           = The array of data (from the df with users removed and maybe
              scaled or with pca)
k_means_arr = An array of fitted kmeans models each corresponding to
              the k=i of the k_range
k_range     = The range of K's to be plotted
best_k      = If the best K_i is known, then this is the value and a red
              circle will be added to the graph at this x-value
OUPUT
fig         = The resulting figure
'''
def plot_kmeans_elbow(X, k_means_arr, k_range, best_k=None):
    centroids = [f.cluster_centers_ for f in k_means_arr]

    #get Euclidean distance from each point to each centroid
    k_euclid = [cdist(X, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]

    #get within-cluster sum of squares
    wcss = [np.sum(d**2) for d in dist]

    #get average intra-cluster distances (absolute values)
    avgWithinSS = [sum(np.abs(d))/X.shape[0] for d in dist]

    #get total sum of squares
    tss = np.sum(pdist(X)**2) / float(X.shape[0])

    #get between cluster sum of squares
    bss = tss - wcss

    # elbow curve - variance
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
    ax1.plot(k_range, bss/tss*100, 'bo-')
    if best_k:
        ax1.plot(k_range[best_k-1], bss[best_k-1]/tss*100, marker='o', \
                            markersize=12, markeredgewidth=2, \
                            markeredgecolor='r', markerfacecolor='None')
    ax1.set_ylim((0,100)) #100%
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Percentage of variance explained (%)')
    ax1.grid(True)

    # elbow curve - intra-cluster dist
    #ax2.plot(k_range, avgWithinSS, 'bo-')
    ax2.plot(k_range, wcss, 'bo-')
    if best_k:
        ax2.plot(k_range[best_k-1], wcss[best_k-1], marker='o', \
                            markersize=12, markeredgewidth=2, \
                            markeredgecolor='r', markerfacecolor='None')
    ax2.set_xlabel('Number of clusters')
    ax2.set_ylabel('Average within-cluster sum of squares')
    ax2.grid(True)

    fig.suptitle('Elbow for KMeans clustering', fontsize=14)
    return fig


