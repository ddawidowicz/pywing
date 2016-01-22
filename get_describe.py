import numpy as np
from scipy import stats


'''
This function prints a basic summary of a numpy array or list that is passed
in. Right now it only works for a vector.
INPUT:
x   = A numpy array or list of values
OUTPUT (prints):
Count
Min
Max
Q1
Median
Q3
Mean
Variance
Skew
Kurtosis
'''
def get_describe(x):
    summary = stats.describe(x)
    N = summary[0]
    minimum = summary[1][0]
    maximum = summary[1][1]
    mu = summary[2]
    var = summary[3]
    skew = summary[4]
    kurtosis = summary[5]

    quartiles = np.percentile(x, [25, 50, 75, 90, 95, 99])
    lower_q = quartiles[0]
    median = quartiles[1]
    upper_q = quartiles[2]
    u90 = quartiles[3]
    u95 = quartiles[4]
    u99 = quartiles[5]

    print '=' * 40
    print '%10s\t%-5d' % ('Count', N)
    print '-' * 40
    print '%10s\t%-5.6f' % ('Min', minimum)
    print '%10s\t%-5.6f' % ('Max', maximum)
    print '%10s\t%-5.6f' % ('Q1', lower_q)
    print '%10s\t%-5.6f' % ('Median', median)
    print '%10s\t%-5.6f' % ('Q3', upper_q)
    print '%10s\t%-5.6f' % ('90th Pct', u90)
    print '%10s\t%-5.6f' % ('95th Pct', u95)
    print '%10s\t%-5.6f' % ('99th Pct', u99)
    print '-' * 40
    print '%10s\t%-5.6f' % ('Mean', mu)
    print '%10s\t%-5.6f' % ('Variance', var)
    print '%10s\t%-5.6f' % ('Skew', skew)
    print '%10s\t%-5.6f' % ('Kurtosis', kurtosis)
    print '=' * 40
    return [N, minimum, maximum, lower_q, median, upper_q, \
            mu, var, skew, kurtosis]


if __name__ == '__main__':
    a = np.array([ 0.68041998,  0.38479833,  0.93520872,  0.71842231,  0.24742548,
            0.69420033,  0.47449312,  0.63102026,  0.60613804,  0.69954211,
            0.74155645,  0.6311252 ,  0.7142314 ,  0.04591061,  0.82005117,
            0.54191001,  0.26640328,  0.26734968,  0.56290822,  0.08784698,
            0.87468249,  0.74891667,  0.49318905,  0.73653535,  0.08849167,
            0.07137012,  0.59092804,  0.66099606,  0.18014941,  0.76283205,
            0.33323582,  0.08020526,  0.3039181 ,  0.30603312,  0.1886784 ,
            0.85797441,  0.2217097 ,  0.36463805,  0.60527096,  0.95181819,
            0.49236651,  0.17142454,  0.72167311,  0.84312013,  0.75648538,
            0.19011191,  0.62614094,  0.39270797,  0.86471821,  0.00207734,
            0.87884829,  0.99029185,  0.14542811,  0.94116029,  0.62883082,
            0.57274534,  0.96922632,  0.64298841,  0.2849718 ,  0.45093078,
            0.73627138,  0.1081515 ,  0.25256071,  0.14714063,  0.30286365,
            0.10820185,  0.27318964,  0.48566335,  0.16098968,  0.1995226 ,
            0.16698878,  0.51763432,  0.0666382 ,  0.9251894 ,  0.92838891,
            0.19592336,  0.3291191 ,  0.04424832,  0.12265539,  0.79244222,
            0.04332742,  0.37734482,  0.80469421,  0.69674545,  0.4298981 ,
            0.00423077,  0.55719898,  0.6283103 ,  0.76796322,  0.32032174,
            0.34548195,  0.92055849,  0.2187969 ,  0.94083845,  0.58542412,
            0.84173231,  0.09482689,  0.76683081,  0.36505303,  0.86605508])
    
    get_describe(a)
