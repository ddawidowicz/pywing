'''
This function takes in a continuous valued data vector and classifies the
observations based on the 10ths decimal value and the 100ths decimal
value. No rounding is performed.
Input:
X       = numpy array, num_observations x num_features, with the data
ref_col     = the column to use for the mapping of value to class
Output:
X_cls   = a new data set with additional columns, one for the class
          based on the 10ths decimal value and one based on the 100ths
          decimal values.
Results are saved in a new file named trn_cls.csv
'''

import numpy as np

def create_class(X, ref_col):
    cls1 = np.array( map(int, X[:,ref_col] % 1 * 10) )
    cls2 = np.array( map(int, X[:,ref_col] % 1 * 100) )
    cls2 = cls2 - (10 * cls1) #subtract off the first decimal place effect

    #cls1 and cls2 are vectors, i.e. cls1.shape = (5000,), and we need
    #to convert them into column matrices, i.e. cls1.shape = (5000,1)
    cls1 = cls1.reshape( -1, 1)
    cls2 = cls2.reshape( -1, 1)
    X_cls = np.hstack([X, cls1, cls2]) 
    np.savetxt('trn_cls.csv', X_cls, delimiter=',')

if __name__ == '__main__':
    infile = '<some_csv>'
    ref_col = 15
    X = np.genfromtxt(infile, delimiter=',')
    create_class(X, ref_col)

