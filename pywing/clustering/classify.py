import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
sns.set(style="darkgrid")


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
n_neigh = The K in the KNN classifier, i.e. the number of neighbors to use
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


'''
This function trains and returns a K-Nearest Neighbors classifier.
INPUTS:
X_trn       = Array of training database
y_trn       = Array of training targets
n_neigh     = Number of neighbors to use
OUTPUT:
neigh       = The trained classifier object
'''
def classify_train(X_trn, y_trn, n_neigh=3):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(X_trn, y_trn)
    return neigh


'''
This function takes a set of features and a set of targets and runs a K-Nearest
Neighbors algorithm using crossvalidation. In this case the target variables
are the cluster labels created by the clustering algorithm.
INPUTS:
X_data      = The data array for training
y_target    = The vector of cluster labels for X_data
n_neigh     = The K in the KNN classifier, i.e. the number of neighbors to use
folds       = The number of folds to use in the crossvalidation process
OUTPUT:
acc         = Scores representing the accuracy of the classifier. It is an
              array the same length as the number of folds provided.
'''
def classify_accuracy(X_data, y_target, title, n_neigh=3, folds=10, tst_size=0.2):
    # Run K-fold crossvalidation to get a baseline accuracy score
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    scores = cross_validation.cross_val_score(neigh, X_data, y_target, cv=folds)

    #Create confusion_matrix plot from a single sample
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_data, \
                                y_target, test_size=tst_size, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = plot_confusion_matrix(cm, y_target, title)
    return [fig, cm, scores]


def plot_confusion_matrix(cm, lbl_vec, title, cmap=plt.cm.Blues):
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(15,8))
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    lbls = ['cluster%d' % i for i in np.unique(lbl_vec)]
    tick_marks = np.arange(len(lbls))
    ax.set_title(title, fontsize=16)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(lbls, rotation=45, fontsize=14)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(lbls, rotation=45, fontsize=14)
    ax.set_xlabel('Predicted label', fontsize=16)
    ax.set_ylabel('True label', fontsize=16)
    cbar = fig.colorbar(cax)
    #cbar.set_ticks([0, 1, 2], update_ticks=True)
    #cbar.set_ticklabels(['Low', 'Medium', 'High'], update_ticks=True)# vertically oriented colorbar
    plt.grid(False)
    plt.tight_layout()
    return fig


def save_classifier(classify_obj, file_name, deg_compress=3):
    joblib.dump(classify_obj, file_name, compress=deg_compress)


def load_classifier(file_name):
    classify_obj = joblib.load(file_name)
    return classify_obj


'''
Currently this does not work for count_cols. I will need to add a second
active_classify_file (for group12) and will need to pull in the params for
the two Gaussians so as to determine for each point, which classifier to use.
It will work for ind_cols and collab_cols.
'''
def classify_new_df(X_tst, main_classify_file, active_classify_file):
    # Load Main and Active classifiers
    main_classify = load_classifier(main_classify_file)
    active_classify = load_classifier(active_classify_file)

    # Predict which main group each sample belongs to
    y_pred = main_classify.predict(X_tst)

    # For actives, predict which subcluster they belong to
    y_pred_11 = active_classify.predict(X_tst[y_pred==1])

    # Distinguish between inactive (0) and active, cluster 0 (now 10)
    # Active cluster 1 is now (11)
    y_pred_11 += 10

    # Add the active cluster designations back into y_pred
    y_pred[y_pred==1] = y_pred_11

    return y_pred


