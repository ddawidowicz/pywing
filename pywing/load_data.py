import numpy as np
from scipy import io
from numpy import genfromtxt 

'''
This function loads data from either a .mat, .csv, or tab 
delimited .txt file. Data is returned as a dictionary
in the case of a .mat file and as a numpy array otherwise
'''
def load_data(infile, mat=False, csv=False, tab=False):
    #loads .mat file with data and returns a dictionary of variables
    #For example, the .mat file may have two variables, X and y
    #and the variables would be accessed like:
    #x_train = mat_file['X']
    #y = mat_file['y']
    if mat:
        mat_file = io.loadmat(infile)
        return mat_file

    
    #load .csv file with data
    elif csv:
        X = genfromtxt(infile, delimiter=',')
        return X

    #load tab delimited text file with data
    elif tab:
        X = genfromtxt(infile, delimiter='\t')
        return X

    else:
        print 'You must specify mat=True or csv=True or tab=True'
        return None


'''
The get_folds() function splits the x and y training set up for n-fold
cross validation training.
INPUTS:
x           = The full training set for the features
y           = The full training set for the targets
num_folds   = The number of folds to use for cross validation
f           = The specific fold we are using for the validation set 
seed_val    = The unique seed value for this run. This ensures that
              all splits will be the same until the seed is changed.
OUTPUTS:
x_trn       = The training portion of x
x_tst       = The testing portion of x specified by f
y_trn       = The training portion of y
y_tst       = The testing portion of y specified by f
'''
def get_folds(x, y, num_folds, f, seed_val):   
    if num_folds <= 1:
        #return x and y as both the training and testing set
        return [x, x, y, y]

    #set up a randomized order for indices based on the number of rows
    #in x and the seed_val passed in
    nrows = x.shape[0]
    idx = range(nrows)
    shuffler = np.random.RandomState(seed_val)
    shuffler.shuffle(idx)

    #create a vector idx2 that has the test fold at the end. For example,
    #if we have a random ordering of range(10) based on the seed_val = 42
    #idx = [8, 1, 5, 0, 7, 2, 9, 4, 3, 6] and we are looking at the third
    #fold from a 5-fold scheme, f=2 (remember folds are 0-indexed), then 
    #we will be using the indices 7 and 2 for the test set. This set of 
    #instructions puts these indices at the end of the list like this,
    #idx = [8, 1, 5, 0, 9, 4, 3, 6, 7, 2]
    tst_size = int( np.floor(nrows / num_folds) )
    start_tst = f * tst_size
    end_tst = (f+1) * tst_size
    idx2 = idx[:start_tst]
    idx2.extend(idx[end_tst:])
    idx2.extend(idx[start_tst:end_tst])
    
    #Using idx2 created above pull out the observations (rows) for the training
    #and testing sets for both x and y (including all columns)
    trn_idx = idx2[:-tst_size]
    tst_idx = idx2[-tst_size:]
    x_trn = x[trn_idx,:]
    x_tst = x[tst_idx,:]
    y_trn = y[trn_idx,:]
    y_tst = y[tst_idx,:]

    return [x_trn, x_tst, y_trn, y_tst]


