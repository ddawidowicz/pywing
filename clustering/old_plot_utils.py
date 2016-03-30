import os
import sys
import pudb
import time
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
#matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import scale
from sklearn.mixture import GMM, DPGMM
from sklearn.ensemble import ExtraTreesClassifier
from scipy.spatial.distance import cdist, pdist
from cluster_usage import *
pd.options.display.width = 200 #set the display width for printing pandas df
np.set_printoptions(linewidth=200) #same for numpy


'''
This function plots the elbow curve for kmeans
INPUTS:
X           = The array of data (from the df with users removed and maybe
              scaled or with pca)
k_means_arr = An array of fitted kmeans models each corresponding to
              the k=i of the k_range
k_range     = The range of k's to be plotted
best_k      = If the best K_i is known, then this is the value and a red
              circle will be added to the graph at this x-value
OUPUT
fig         = The resulting figure
'''
def plot_kmeans_elbox(X, k_means_arr, k_range, best_k=None):
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
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(k_range, bss/tss*100, 'bo-')
    if best_k:
        ax1.plot(k_range[best_k-1], bss[best_k-1]/tss*100, marker='o', \
                            markersize=12, markeredgewidth=2, \
                            markeredgecolor='r', markerfacecolor='None')
    ax1.set_ylim((0,100))
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


'''
This function plots graphs for the gap statistic method of finding the number
of clusters K.
INPUTS:
k_range     = An array of the possible K values
logWks      = An array of the log(W_k) values for each value of K=k_i. The W_k
              value is a sum across clusters of the sum of squared distances
              between each data point and the centroid of its assigned cluster.
logWkbs     = An array of the log(Wks) values for the reference distribution. 
              The reference distribution is a uniform sampling over the 
              bounding box.
sk          = An array of the sk values for each value of K=k_i. The sk value
              is \sqrt{1 + \frac{1}{B}} * std(Wkbs)
OUTPUT:
fig         = The resulting panel of plots for the gap statistic
'''
def plot_kmeans_gap(k_range, logWks, logWkbs, sk):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,12))
    fig.subplots_adjust(hspace=0.3)

    #Sum of all inter-cluster distance per K
    ax1.plot(k_range, np.exp(logWks), 'mo-')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Intra-cluster distances ($W_k$)')
    ax1.grid()

    #Plot the two curves log(Wks) and log(Wkbs) to visually see gap
    ax2.plot(k_range, logWks, 'bo-', label='logWks')
    ax2.plot(k_range, logWkbs, 'ro-', label='logWkbs')
    ax2.set_xlabel('Number of clusters')
    ax2.legend(loc=0)
    ax2.grid()
    
    #Explicitly plot the gap between log(Wks) and log(Wkbs)
    ax3.plot(k_range, (logWkbs-logWks), 'mo-')
    ax3.set_xlabel('Number of clusters')
    ax3.set_ylabel('$Gap(k)$')
    ax3.set_title('Look for K with greatest y-value')
    ax3.grid()
    
    #Plot the optimal number of clusters, i.e. the smallest k such that 
    #\mathrm{Gap}(k) \geq \mathrm{Gap}(k+1) - s_{k+1}
    #Look for the first K that gives a positive value in the plot below
    gap_k = (logWkbs-logWks)[:-1]
    gap_k1 = (logWkbs-logWks)[1:]
    sk1 = sk[1:]
    y = gap_k - (gap_k1 - sk1)
    ax4.bar(k_range[:-1], y)
    ax4.set_xlabel('Number of clusters')
    ax4.set_ylabel('$Gap(k)-(Gap(k+1)-s_{k+1})$')
    ax4.set_title('Look for first K that gives a positive value')
    ax4.grid()
    
    fig.suptitle("Number of clusters (Gap Method)", fontsize=14)
    return fig


'''
This function plots the silhouette score and cluster membership for a range of 
the number of clusters. It will run either with the KMeans or DPGMM clustering
models.
INPUTS:
X_t         = The array of data features
n_clustesrs = The number of clusters to evaluate
use_kmeans  = Whether to use KMeans - if false, it will use DPGMM
alpha       = The alpha value for DPGMM
NOTE: there are other run options at the very top of the function
OUTPUT:
fig         = The resulting figure, call plt.show() after running this function
'''
def plot_silhouette(X_t, n_clusters, use_kmeans=True, alpha=0.5):
    cv_type = 'diag'
    iters = 30
    n_init = 30

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([0.0, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_t) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    if use_kmeans:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X_t)
    else:
        [cluster_labels, prob_vec, bic_dpgmm, log_prob_dpgmm] = \
                    get_best_dpgmm(X_t, n_clusters, cv_type, alpha, iters, n_init, rand_state=10)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_t, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    print(pd.Series(cluster_labels).value_counts())
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_t, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    return fig


'''
This function creates a plot showing the impact features in X have on 
predicting the cluster labels
INPUTS:
X		= The array created from the numeric fields of df_f_all
lbl_vec	= The vector of cluster labels based on X
x_cols	= The column names of X, viz. df_f_all.iloc[:,1:-6].columns
f_cols	= The columns (features) we want to assess
OUTPUTS:
fig		= The resulting plot
Note that this function will aslo print out the influence weights
for each feature in f_cols
'''
def plot_feature_importance(X, lbl_vec, x_cols, f_cols):
	#select only the cols we care about, f_cols
	X_tmp = np.array(pd.DataFrame(X, columns=x_cols)[f_cols])
	
	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=250,
								  random_state=0)
	forest.fit(X_tmp, lbl_vec)
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
				 axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(len(f_cols)):
		print("%d. %s (%f)" % (f + 1, f_cols[indices[f]], \
                                            importances[indices[f]]))

	# Plot the feature importances of the forest
	fig, ax = plt.subplots(figsize=(16,12))
	ax.bar(range(len(f_cols)), importances[indices],
		   color="r", yerr=std[indices], align="center")
	ax.set_title("Feature importances (HS Students, bin2)", fontsize=18)
	ax.set_xlim([-1, len(f_cols)])
	new_lbls = []
	for i in indices:
		new_lbls.append(f_cols[i])
	ax.set_xticks(np.arange(len(f_cols))-0.25)
	ax.set_xticklabels(new_lbls, fontsize=14, rotation='45')
	ax.set_ylabel('Influence weight (percent)', fontsize=14)
	plt.tick_params(axis='both', which='major', labelsize=14)
	return fig
	

# ==================================== GMM ================================== #
def plot_num_components_gmm(X):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 11)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = GMM(n_components=n_components, covariance_type=cv_type, \
                                                    n_init=30, n_iter=30)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['k', 'r', 'g', 'b', 'c', 'm', 'y'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    fig, ax = plt.subplots(figsize=(10,8))
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(ax.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    ax.set_xticks(n_components_range)
    ax.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    ax.set_title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    ax.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    ax.set_xlabel('Number of components')
    ax.legend([b[0] for b in bars], cv_types)
    return [fig, lowest_bic]
    
    
def plot_num_restarts_gmm(X, num_c, cv_type, iters, max_inits):
    bic = []
    for num_init in np.arange(1,max_inits):
        gmm = GMM(n_components=num_c, covariance_type=cv_type, \
                                    n_iter=iters, n_init=num_init)
        gmm.fit(X)
        bic.append(gmm.bic(X))
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(np.arange(1, max_inits), bic)
    ax.set_title('BIC vs. Number Restarts GMM')
    ax.set_xlabel('Number of restarts')
    ax.set_ylabel('BIC score')
    return fig
    

def plot_num_iters_gmm(X, num_c, cv_type, max_iters, n_init):
    bic = []
    for iters in np.arange(1, max_iters):
        gmm = GMM(n_components=num_c, covariance_type=cv_type, \
                                        n_iter=iters, n_init=n_init)
        gmm.fit(X)
        bic.append(gmm.bic(X))
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(np.arange(1, max_iters), bic)
    ax.set_title('BIC vs. Number Iterations GMM')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('BIC score')
    return fig
        
        
# =================================== DPGMM ================================= #
def plot_num_components_dpgmm(X, max_num_c, cv_type, alpha, iters, n_init):
    bic = []
    for comp in np.arange(1, max_num_c):
        best_bic = np.inf
        for j in xrange(n_init):
            dpgmm = DPGMM(n_components=comp, covariance_type=cv_type, \
                                            alpha=alpha, n_iter=iters)
            dpgmm.fit(X)
            b = dpgmm.bic(X)
            if b < best_bic:
                best_bic = b
        bic.append(best_bic)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(np.arange(1, max_num_c), bic)
    ax.set_title('BIC vs. Number Components DPGMM')
    ax.set_xlabel('Number of components')
    ax.set_ylabel('BIC score')
    return fig


def plot_alpha_dpgmm(X, num_c, cv_type, alphas, iters, n_init):
    bic = []
    for a in alphas:
        best_bic = np.inf
        for j in xrange(n_init):
            dpgmm = DPGMM(n_components=num_c, covariance_type=cv_type, \
                                                alpha=a, n_iter=iters)
            dpgmm.fit(X)
            b = dpgmm.bic(X)
            if b < best_bic:
                best_bic = b
        bic.append(best_bic)

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(alphas, bic, 'bo-', lw=2)
    ax.set_title('BIC vs. Alpha DPGMM')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('BIC score')
    return fig


def plot_num_iters_dpgmm(X, num_c, cv_type, alpha, max_iters, n_init):
    bic = []
    for iters in np.arange(1, max_iters):
        best_bic = np.inf
        for j in xrange(n_init):
            dpgmm = DPGMM(n_components=comp, covariance_type=cv_type, \
                                                alpha=a, n_iter=iters)
            dpgmm.fit(X)
            b = dpgmm.bic(X)
            if b < best_bic:
                best_bic = b
        bic.append(best_bic)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(np.arange(1, max_iters), bic)
    ax.set_title('BIC vs. Number of Iterations DPGMM')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('BIC score')
    return fig
    
    
def plot_cluster_metrics(df_log):
    ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = df_log.hist(layout=(4, 2), \
                                                        figsize=(12,8))
    #ax1.set_xlim([0,5])
    #ax2.set_xlim([0,3])
    #ax3.set_xlim([0,13])
    #ax4.set_xlim([0,4])
    #ax5.set_xlim([0,5])
    #ax6.set_xlim([0,5])

    new_lbls1 = ['{:,.0f}'.format(np.exp(item)) for item in ax1.get_xticks()]
    ax1.set_xticklabels(new_lbls1)
    new_lbls2 = ['{:,.0f}'.format(np.exp(item)) for item in ax2.get_xticks()]
    ax2.set_xticklabels(new_lbls2)
    new_lbls3 = ['{:,.0f}'.format(np.exp(item)) for item in ax3.get_xticks()]
    ax3.set_xticklabels(new_lbls3)
    new_lbls4 = ['{:,.0f}'.format(np.exp(item)) for item in ax4.get_xticks()]
    ax4.set_xticklabels(new_lbls4)
    new_lbls5 = ['{:,.0f}'.format(np.exp(item)) for item in ax5.get_xticks()]
    ax5.set_xticklabels(new_lbls5)
    new_lbls6 = ['{:,.0f}'.format(np.exp(item)) for item in ax6.get_xticks()]
    ax6.set_xticklabels(new_lbls6)
    new_lbls7 = ['{:,.0f}'.format(np.exp(item)) for item in ax7.get_xticks()]
    ax7.set_xticklabels(new_lbls7)
    #new_lbls8 = ['{:,.0f}'.format(np.exp(item)) for item in ax8.get_xticks()]
    #ax8.set_xticklabels(new_lbls8)
    return plt


'''
This function plots clusters that have been formed using 2 PCA components.
INPUTS:
X_t     = A numpy array of the numeric features from df_f_all (n_usr x 86)
lbl_vec = A numpy array of the cluster labels for each row in X_t
K       = The number of clusters (either 3 or 4)
OUTPUT:
fig     = The resulting figure
'''
def plot_2d_clusters(X_t, lbl_vec, K):
    fig, ax = plt.subplots()
    ax.plot(X_t[lbl_vec==0][:,0], X_t[lbl_vec==0][:,1], 'bo', label='cluster1')
    ax.plot(X_t[lbl_vec==1][:,0], X_t[lbl_vec==1][:,1], 'ro', label='cluster2')
    ax.plot(X_t[lbl_vec==2][:,0], X_t[lbl_vec==2][:,1], 'co', label='cluster3')
    if K==4:
        ax.plot(X_t[lbl_vec==3][:,0], X_t[lbl_vec==3][:,1], 'mo', \
                                                    label='cluster4')
    ax.set_title('Resulting Clusters')
    ax.set_xlabel('X component')
    ax.set_ylabel('Y component')
    return fig


'''
This function plots clusters that have been formed using 3 PCA components.
INPUTS:
X_t     = A numpy array of the numeric features from df_f_all (n_usr x 86)
lbl_vec = A numpy array of the cluster labels for each row in X_t
K       = The number of clusters (either 3 or 4)
OUTPUT:
fig     = The resulting figure
'''
def plot_3d_clusters(X_t, lbl_vec, K):
    #plt.rcParams['figure.figsize'] = 16, 8
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_t[lbl_vec==0][:,0], X_t[lbl_vec==0][:,1], \
            X_t[lbl_vec==0][:,2], 'bo', label='cluster1')
    ax.plot(X_t[lbl_vec==1][:,0], X_t[lbl_vec==1][:,1], \
            X_t[lbl_vec==1][:,2], 'ro', label='cluster2')
    ax.plot(X_t[lbl_vec==2][:,0], X_t[lbl_vec==2][:,1], \
            X_t[lbl_vec==2][:,2], 'co', label='cluster3')
    if K==4:
        ax.plot(X_t[lbl_vec==3][:,0], X_t[lbl_vec==3][:,1], \
                X_t[lbl_vec==3][:,2], 'co', label='cluster4')
    ax.set_title('Resulting Clusters')
    ax.set_xlabel('X component')
    ax.set_ylabel('Y component')
    ax.set_zlabel('Z component')
    return fig


'''
This function plots a pie chart for the cluster membership within a dataset.
INPUTS:
data    = The 4x1 vector of counts within each category. They must have been
          ordered to correspond to ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']
t       = A title for the plot
OUTPUT:
fig     = The resulting plot
'''
def plot_cluster_pie(data, t):
    # ------------------------------- Set up --------------------------------- #
    lbls = ['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4']
    colors = ['#125df2', 'magenta', 'cyan', 'red']
    explode = [0.0, 0.0, 0.0, 0.0]

    # ------------------------------ Set Font -------------------------------- #
    #mpl.rcParams['text.color'] = 'white'
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = '16'

    # ------------------------------ Plot it --------------------------------- #
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.pie(data, explode=explode, colors=colors, \
    autopct='%1.1f%%', shadow=True, startangle=90, counterclock=False)
    ax.axis('equal')
    ax.set_title(t, fontsize=18)

    # --------------------- Reset Defaults for Legend ------------------------ #
    #mpl.rcdefaults()
    #mpl.rcParams['font.size'] = '14'
    ax.legend(labels=lbls, loc=(1.0, 0.03), shadow=False, fontsize=14)

    return fig

