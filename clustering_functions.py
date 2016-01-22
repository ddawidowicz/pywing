'''
This set of functions are a set of functions I found useful in creating 
clusters of students and teachers. The first three sections, ELBOW METHOD, 
GAP STATISTIC, and Alternative to Gap Stat are used to determine the number of 
clusters. Depending on the sample size these methods gave varying degrees of 
convincing number of clusters. I also used the plot_silhouette() function to 
evaluate the number of clusters chosen.

The next 3 sections KMEANS, DPGMM, and GMM show how to run different clustering
algorithms. I found KMEANS to be the quickest and easiest to work with for
prototyping. The GMM algorithm provides a probability distribution over clusters, 
which can be useful and the DPGMM allows for a dynamic determination of the
number of clusters.

The next sections, SILHOUETTE SCORE, presents another method for determining
the number of clusters and for scoring the clusters. The SILHOUETTE score
looks at the intra-cluster and inter-cluster distances to score the goodness of
the clustering. 

The final section, CLASSIFIER, shows how to use a K-Nearest Neighbors 
classifier to classify new users to a cluster from an already trained clustering
model. I found that it was around 98-99% accurate.

GENERAL NOTES:
In general clustering should be performed on scaled data and I found that using
PCA first was very helpful. I used 2-3 components because it was easy to 
visualize, but using more will capture more of the variance. Here is an example
of scaling and using PCA:
X = scale(np.array(df_f, dtype=float)) 
X = PCA(n_components=2).fit_transform(X)
'''


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import linalg
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.mixture import GMM, DPGMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KNeighborsClassifier


# =============================== ELBOW METHOD =============================== #
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
OUTPUT:
fig         = The resulting figure

EXAMPLE CALL:
k_range = np.arange(1, max_clusters+1)
k_means_arr = [KMeans(n_clusters=k).fit(X) for k in k_range]
fig = plot_kmeans_elbox(X, k_means_arr, k_range, best_k=None)
plt.show()
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


# =============================== GAP STATISTIC ============================== #
# See https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
# for more on gap statistic

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


'''
This function plots graphs for the gap statistic method of finding the number
of clusters K. Run get_gap_stats() first.
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


# =========================== Alternative to Gap Stat ======================== #
'''
This method uses the ratio between the distortion of the actual clustering to the 
distortion of an estimated null reference distribution (uniform distribution). The
lower the ratio, the higher chance of the presence of clusters in the data. Because
of the recursive nature of the algorithm, I found that I had to use memoization or
the algorithm would crash with more than 16 clusters.
Adapted from https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
Paper: http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf
'''

'''
This function allows for memoization (caching) of results for the recursive 
computation of the weighting value a. It is used as a decorator below.
INPUT:
func    = The function that will be called
OUTPUT:
helper  = The memoized function
Adapted from
http://www.python-course.eu/python3_memoization.php 
and
http://people.ucsc.edu/~abrsvn/NLTK_parsing_demos.pdf
For a more complete version of memoization see
https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
'''
def memoize(func):
    memo = {}
    def helper(*args):
        if args not in memo:            
            memo[args] = func(*args)
        return memo[args]
    return helper


'''
This function computes the weighting constant used to account for the
data dimensionality. It has a memoize decorator to help with the
recursive nature of the function - see above.
INPUT:
k     = The current value of K (num_clusters) being evaluated
Nd    = The number of features in the data
OUTPUT:
a     = The weighting factor a
'''
@memoize
def get_a(k, Nd):
    if k == 2:
        a = 1.0 - 3.0/(4.0 * Nd) 
    else:
        a = get_a(k-1, Nd) + (1.0 - get_a(k-1, Nd))/6.0
    return a


'''
This function is an implementation of the method described in the paper 
referenced above to find the ratios needed for the plot.
INPUTS:
X     = The data array (with features along columns)
k     = The specific k (num_clusters) being evaluated
Skm1  = The sum of distortions (S_(k-1)) for k-1
OUTPUTS:
fs    = The ratio we need to determine the number of clusters
Sk    = The sum of distortions needed to compute k+1
'''
def f_K(X, k, Skm1=0):
    n_init = 10
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(X)
    Sk = kmeans.inertia_
    Nd = X.shape[1]
    #was this below, changed it to call this memoized get_a() function
    #a = lambda k, Nd: 1 - 3/(4*Nd) if k == 2 else a(k-1, Nd) + (1-a(k-1, Nd))/6
    
    if k == 1:
        fs = 1
    elif Skm1 == 0:
        fs = 1
    else:
        fs = Sk/(get_a(k,Nd)*Skm1)
    return fs, Sk


'''
The wrapper function that gets called to return the array of ratios
needed to determine K.
INPUTS:
X            = The data array (with features along columns)
max_clusters = The maximum number of clusters to consider
OUTPUTS:
k_range      = A numpy array from 1-max_clusters (inclusive)
fs           = A numpy array of the ratios needed to compute K
'''
def get_K_fk(X, max_clusters=15):
    k_range = np.arange(1, max_clusters+1)
    fs = np.zeros(max_clusters)
    fs[0], Sk = f_K(X, k_range[0])
    for k in k_range[1:]:
        fs[k-1], Sk = f_K(X, k, Skm1=Sk)
    return k_range, fs


'''
This function plots the results of the get_K_fk() function
You want to look for the lowest point. There is some evidence
from the paper that you need to get below 0.88 or so to be
meaningful. I set a break at 0.85 to highlight if 1 cluster
might be the best option.
INPUTS:
k_range      = A numpy array from 1-max_clusters (inclusive)
fs           = A numpy array of the ratios needed to compute K
OUTPUT:
fig          = The completed figure
'''
def plot_f_K(k_range, fs):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(k_range, fs, 'bo-', alpha=0.9)
    ax.set_xlabel('Number of clusters K', fontsize=16)
    ax.set_ylabel('f(K)', fontsize=16) 
    ax.set_ylim(0, 1.2)
    min_fs = np.min(fs)
    foundfK = np.where(fs == min_fs)[0][0] + 1
    if min_fs < 0.85:
        t = 'f(K) finds %d clusters\n(Look for lowest point)' % (foundfK)
    else:
        t = 'f(K) finds 1 or maybe %d clusters\n(Look for lowest point)' % (foundfK)
    ax.set_title(t, fontsize=16)
    return fig


# =================================== KMEANS ================================= #
'''
KMEANS can be run with:
K = 4 #number of clusters
n_init = 100 #number of restarts
kmeans = KMeans(n_clusters=K, n_init=n_init)
kmeans.fit(X_t)
lbl_vec = kmeans.predict(X_t)
'''

# =================================== DPGMM ================================= #
# Example of using Dirichlet Process Gaussian Mixture Models
'''
This function shows an example of calling the Dirichlet Process Gaussian
Mixture Model clustering algorithm. The DPGMM algorithm is useful because
the number of clusters is in part determined by the algorithm itself.
I created this wrapper because I wanted multiple restarts of the algorithm. 
This is slower than calling the function directly, but this shows how to do 
that also. See also, 
http://scikit-learn.org/stable/modules/generated/sklearn.mixture.DPGMM.html
INPUTS:
X           = Data array
num_c       = Best guess as to the number of clusters
cv_type     = Covariance parameter to use, one of 'spherical', 'tied', 'diag', 'full'
alpha       = Concentration parameter of the Dirichlet process - higher alpha
              means more clusters - expected number of clusters is alpha*log(n)
iters       = Max number of iterations for algorithm to converge
n_init      = How many restarts to perform

OUTPUTS:
lbl_vec         = A cluster label for each row of X (0-indexed)
prob_vec        = Probability of cluster membership
bic_dpgmm       = The BIC score
log_prob_dpgmm  = Log probability (a scoring metric)

EXAMPLE CALL:
num_c = 4 
cv_type = 'diag'
alpha = 7
iters = 30
n_init = 50
[lbl_vec, prob_vec, bic_dpgmm, log_prob_dpgmm] = \
                get_best_dpgmm(X, num_c, cv_type, alpha, iters, n_init)
'''
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


# ==================================== GMM =================================== #
'''
This function shows how to call the Gauassian Mixture Model clustering algorithm.
INPUTS:
X           = Data array
num_c       = Best guess as to the number of clusters
cv_type     = Covariance parameter to use, one of 'spherical', 'tied', 'diag', 'full'
iters       = Max number of iterations for algorithm to converge
num_init    = How many restarts to perform

OUTPUTS:
gmm         = The Gaussian Mixture Model clustering trained object
lbl_vec         = A cluster label for each row of X (0-indexed)
prob_vec        = Probability of cluster membership

EXAMPLE CALL:
num_c = 4
cv_type = 'diag'
iters = 30
num_init = 30
[gmm, lbl_vec, prob_vec] = run_gmm_clusters(X, num_c, cv_type, iters, num_init)
print 'BIC =', gmm.bic(X)
print 'Log prob = ', np.sum(gmm.score(X))
print pd.Series(lbl_vec).value_counts()
'''
def run_gmm_clusters(X, num_c, cv_type, iters, num_init):
    gmm = GMM(n_components=num_c, covariance_type=cv_type, \
                                n_iter=iters, n_init=num_init)
    gmm.fit(X)
    lbl_vec = gmm.predict(X)
    prob_vec = gmm.predict_proba(X)
    return [gmm, lbl_vec, prob_vec]


# ============================== Silhouette Score ============================ #
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


# =============================== CLASSIFIER ================================= #
'''
This function takes a training set and a testing set and classifies the data
in the testing set based on a classifier trained on the training set. It is 
used after a clustering algorithm has been trained and run on a set of data
(the training set) to classify a new set of data (the testing set). The
algorithm uses the K-Nearest Neighbors algorithm.
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
       
       

