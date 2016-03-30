import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
sns.set(style="darkgrid")


'''
This function creates a plot showing the impact features in X have on
predicting the cluster labels
INPUTS:
X       = The array created from the numeric fields of df_f_all
lbl_vec = The vector of cluster labels based on X
x_cols  = The column names of X, viz. df_f_all.iloc[:,1:-6].columns
f_cols  = The columns (features) we want to assess
OUTPUTS:
fig     = The resulting plot
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
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(range(len(f_cols)), importances[indices],
           color="r", yerr=std[indices], align="center")
    ax.set_title("Feature importance (demographic clusters)", fontsize=18)
    ax.set_xlim([-1, len(f_cols)])
    new_lbls = []
    for i in indices:
        new_lbls.append(f_cols[i])
    ax.set_xticks(np.arange(len(f_cols))-0.25)
    ax.set_xticklabels(new_lbls, fontsize=14, rotation='45')
    ax.set_ylabel('Influence weight (percent)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    return fig


def print_feature_import(df_use, X, cols_to_use, pca):
    coef = pca.components_.T
    cols = list(df_use[cols_to_use].columns)
    if X.shape[1] == 2:
        print pd.DataFrame(coef, columns=['PCA-1', 'PCA-2'], index=cols) #gives table of column import
    elif X.shape == 3:
        print pd.DataFrame(coef, columns=['PCA-1', 'PCA-2', 'PCA-3'], index=cols) #gives table of column import
    else:
        print 'too many components'

def print_cluster_mean_med(user_cluster_dfs, cols_to_use, group):
    if group in [11, 12]:
        print '%s | %s | %s | %s | %s | %s | %s | %s | %s' % \
        ('Column', 'mean c0', 'median c0', 'mean c1', 'median c1', 'mean c2', 'median c2', 'mean rm', 'median rm')
        print '------- | ------- | --------- | ------- | --------- | ------- | --------- | ------- | --------- '
    elif group == 'all':
        print '%s | %s | %s | %s | %s | %s | %s' % \
        ('Column', 'Inact mean', 'Inact median', 'Act mean', 'Act median', 'Extrm mean', 'Extrm median')
        print '------- | ------- | --------- | ------- | --------- | ------- | ---------'
    else:
        print 'Incorrect group'
        return
    for idx in xrange(len(cols_to_use)):
        if len(user_cluster_dfs) == 2:
            print '%s | %.5f | %.2f | %.5f | %.2f ' % \
            (cols_to_use[idx], \
            user_cluster_dfs[0].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[0].describe().loc['50%', cols_to_use[idx]], \
            user_cluster_dfs[1].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[1].describe().loc['50%', cols_to_use[idx]])
        elif len(user_cluster_dfs) == 3:
            print '%s | %.5f | %.2f | %.5f | %.2f | %.5f | %.2f' % \
            (cols_to_use[idx], \
            user_cluster_dfs[0].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[0].describe().loc['50%', cols_to_use[idx]], \
            user_cluster_dfs[1].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[1].describe().loc['50%', cols_to_use[idx]], \
            user_cluster_dfs[2].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[2].describe().loc['50%', cols_to_use[idx]])
        elif len(user_cluster_dfs) == 4:
            print '%s | %.5f | %.2f | %.5f | %.2f | %.5f | %.2f | %.5f | %.2f ' % \
            (cols_to_use[idx], \
            user_cluster_dfs[0].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[0].describe().loc['50%', cols_to_use[idx]], \
            user_cluster_dfs[1].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[1].describe().loc['50%', cols_to_use[idx]], \
            user_cluster_dfs[2].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[2].describe().loc['50%', cols_to_use[idx]], \
            user_cluster_dfs[3].describe().loc['mean', cols_to_use[idx]], \
            user_cluster_dfs[3].describe().loc['50%', cols_to_use[idx]])
        else:
            print 'no support for over 4 clusters'


