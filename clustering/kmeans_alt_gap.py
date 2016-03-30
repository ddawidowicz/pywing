import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set(style="darkgrid")


'''
This method uses the ratio between the distortion of the actual clustering to
the distortion of an estimated null reference distribution (uniform
distribution). The lower the ratio, the higher chance of the presence of
clusters in the data. Because of the recursive nature of the algorithm, I found
that I had to use memoization or the algorithm would crash with more than
16 clusters.
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
This function gets the weighting constant used to normalize for the
data dimensionality. It has a memoize decorator to help with the
recursive nature of the function - see above.
INPUT:
k     = The current value of K (num_clusters) being evaluated
Nd    = The number of columns (features / dimesions) of the data array
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
def plot_alt_gap(k_range, fs):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(k_range, fs, 'bo-', alpha=0.9)
    ax.set_xlabel('Number of clusters K', fontsize=16)
    ax.set_ylabel('f(K)', fontsize=16)
    ax.set_ylim(0, 1.1) #1.1 to show 100%
    ax.set_xticks(k_range)
    ax.set_yticks(np.arange(0,1.2,0.1)) #1.1 to show 100%
    min_fs = np.min(fs)
    foundfK = np.where(fs == min_fs)[0][0] + 1
    if min_fs < 0.85:
        t = 'f(K) finds %d clusters\n(Look for lowest point)' % (foundfK)
    else:
        #t = 'f(K) finds 1 or maybe %d clusters\n(Look for lowest point)' % (foundfK)
        t = 'f(K) finds %d clusters\n(Look for lowest point)' % (foundfK)
    ax.set_title(t, fontsize=16)
    ax.grid(True)
    return fig


