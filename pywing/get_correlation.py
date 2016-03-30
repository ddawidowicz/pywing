'''
This function examines the correlation between all columns of a dataset and the
specified column using the Pearson correlation. 
Results are printed and returned.
Inputs:
infile  = data file to use
ref_col = the column against which the other columns correlations are measured.

Outputs:
correlations = a list of lists sorted (descending) by strength of correlation.
Each inner list = [col_number, correlation, p-value]
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
from load_data import load_data



'''
This function plots a correlation matrix for the features identified in keepers
INPUTS:
df_k        = The original df_f feature df with survival != inf or NaN
              and with only the columns listed in keepers[1:]
keepers     = A list of columns to keep for this step, from keepers[1:] in
              the control panel
scale_it    = A boolean that determines if the data should be z-scaled
OUTPUT:
fig         = The resulting figure with the correlation matrix for the
              columns given in keepers
'''
def get_feature_corr(df_k, keepers, scale_it=True):
    if scale_it:
        X_k = scale(np.array(df_k, dtype=float))
    else:
        X_k = np.array(df_k, dtype=float)
    df_xk = pd.DataFrame(X_k)
    df_xk.columns = keepers

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(figsize=(15, 15))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.corrplot(df_xk, annot=True, sig_stars=True,
                 diag_names=False, cmap=cmap, ax=ax)
    ax.set_title('Correlation between training features')
    fig.tight_layout()
    return fig
    
    
def get_corr(infile, ref_col):
    # --------------------------- Set File Type ------------------------------ #
    mat = False
    csv = False
    tab = False
    if infile[-3:] == 'mat':
        mat = True
    elif infile[-3:] == 'csv':
        csv = True
    elif infile[-3:] == 'tsv':
        tab = True
    else:
        csv = True #for now default is csv, but could print error msg to user


    # ------------------------------- Get Data ------------------------------- #
    X = load_data(infile, mat, csv, tab)
    #If X is a vector, i.e. X.shape = (5000,), then we need
    #to convert it into a matrix, i.e. X.shape = (5000,1)
    try:
        X.shape[1] #check if this is valid
    except:
        X = X.reshape( -1, 1)
    [n_row, n_col] = X.shape
    front = X[:, range(ref_col)] #every column up to (not incld) ref_col
    back = X[:, range((ref_col+1), n_col)] #every col after (not incld) ref_col
    a = np.hstack([front, back]) #all columns except ref_col (all rows)
    b = X[:,ref_col] #all rows, only ref_col
    
    
    # -------------------------- Get Correlations ---------------------------- #
    correlations = []
    for i in xrange(a.shape[1]):
        [corr, p] = pearsonr(a[:,i], b)
        #print 'Feature %d =\t%7.4f\t(p=%.6f)' % (i, corr, p)
        correlations.append([i, corr, p])

    correlations.sort(key = lambda x: np.abs(x[1]), reverse=True) #descending
    for r in correlations:
        print 'Feature %d =\t%7.4f\t(p=%.6f)' % (r[0], r[1], r[2])

    return correlations


if __name__ == '__main__':
    # ========================= Control Panel ================================ #
    infile = 'data/dc1_trn_num5k.csv'
    ref_col = 15
    # ======================================================================== #
    get_corr(infile, ref_col)
