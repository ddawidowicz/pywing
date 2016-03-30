import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set(style="darkgrid")

'''
This function plots clusters that have been formed using 2 PCA components.
INPUTS:
X_t     = A numpy array of the numeric features from df_f_all (n_usr x 86)
lbl_vec = A numpy array of the cluster labels for each row in X_t
K       = The number of clusters (either 3 or 4)
markers = A list of the colors and marker style to use - defaults to only 6
          colors - will need to pass in if K > 6.
OUTPUT:
fig     = The resulting figure
'''
def plot_2d_clusters(X, lbl_vec, K, markers=['bo','ro','co','mo','ko','go']):
    fig, ax = plt.subplots(figsize=(15,8))
    for idx in xrange(K):
        prct = 100.0 * len(lbl_vec[lbl_vec==idx]) / float(len(lbl_vec))
        tmp_lbl = 'cluster%d (%.1f%%)' % (idx, prct)
        ax.plot(X[lbl_vec==idx][:,0], X[lbl_vec==idx][:,1], markers[idx], \
                                    alpha=0.75, label=tmp_lbl)
    ax.set_title('Resulting Clusters', fontsize=16)
    ax.set_xlabel('X component', fontsize=14)
    ax.set_ylabel('Y component', fontsize=14)
    ax.legend(loc=0, labelspacing=1.5, fontsize=14)
    return fig


def plot_2d_clusters_all(X, lbl_vec):
    fig, ax = plt.subplots(figsize=(15,8))
    prct0 = 100.0 * lbl_vec[lbl_vec==0].shape[0] / float(lbl_vec.shape[0])
    prct1 = 100.0 * lbl_vec[lbl_vec==1].shape[0] / float(lbl_vec.shape[0])
    prct2 = 100.0 * lbl_vec[lbl_vec==2].shape[0] / float(lbl_vec.shape[0])

    ax.scatter(X[lbl_vec==1][:,0], X[lbl_vec==1][:,1], c='b', label='Active (%.2f%%)' % prct1)
    ax.scatter(X[lbl_vec==2][:,0], X[lbl_vec==2][:,1], c='c', label='Extreme (%.2f%%)' % prct2)
    ax.scatter(X[lbl_vec==0][:,0], X[lbl_vec==0][:,1], s=50, c='r', label='Inactive (%.2f%%)' % prct0)
    ax.set_title('Resulting Clusters (3 main groups)', fontsize=16)
    ax.set_xlabel('X component', fontsize=14)
    ax.set_ylabel('Y component', fontsize=14)
    ax.legend(loc=2, labelspacing=1.5, fontsize=14)

    return fig


'''
This function plots clusters that have been formed using 3 PCA components.
INPUTS:
X_t     = A numpy array of the numeric features from df_f_all (n_usr x 59)
lbl_vec = A numpy array of the cluster labels for each row in X_t
K       = The number of clusters (either 3 or 4)
OUTPUT:
fig     = The resulting figure
'''
def plot_3d_clusters(X, lbl_vec, K, markers=['bo','ro','co','mo','ko','go']):
    fig = plt.figure(figsize=(15,25))
    ax1 = fig.add_subplot(411, projection='3d')
    ax2 = fig.add_subplot(412, projection='3d')
    ax3 = fig.add_subplot(413, projection='3d')
    ax4 = fig.add_subplot(414, projection='3d')
    fig.subplots_adjust(hspace=0.1)
    axes_list = [ax1, ax2, ax3, ax4]
    #(x/z)=(15, 75),(15,-100); (x/y)=(100,85); (y/z)=(10, -175); (other)=(45, -110), (30 100)
    rotation_list = [(15,75), (15,-100), (100,85), (10,-175)]

    for ax,rot in zip(axes_list,rotation_list):
        for idx in xrange(K):
            prct = 100.0 * len(lbl_vec[lbl_vec==idx]) / float(len(lbl_vec))
            tmp_lbl = 'cluster%d (%.1f%%)' % (idx, prct)
            ax.plot(X[lbl_vec==idx][:,0], \
                    X[lbl_vec==idx][:,1], \
                    X[lbl_vec==idx][:,2], \
                    markers[idx], \
                    alpha=0.75, \
                    label=tmp_lbl)
        ax.set_title('Resulting Clusters')
        ax.set_xlabel('X component')
        ax.set_ylabel('Y component')
        ax.set_zlabel('Z component')
        ax.view_init(elev=rot[0], azim=rot[1])

    return fig


