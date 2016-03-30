'''
Plot function may not be working
'''

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
sns.set(style="darkgrid")

'''
This function computes the silhouette score for the clusters
The silhouette_score gives the average value for all the samples. This gives a
sense of the density and separation of the formed clusters. Scores range from
[-1, 1], where the best value is 1 and the worst value is -1. Values near 0
indicate overlapping clusters. Negative values generally indicate that a
sample has been assigned to the wrong cluster, as a different cluster is
more similar.
INPUTS:
X_t         = The array of data features
n_clustesrs = The number of clusters to evaluate
OUTPUT:
score       = The silhouette score
'''
def get_silhouette_score(X, cluster_labels):
    silhouette_avg = silhouette_score(X, cluster_labels)
    return silhouette_avg


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
    fig, ax1 = plt.subplots(figsize=(15, 8))

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


