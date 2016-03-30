import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set(style="darkgrid")


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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,8))
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


