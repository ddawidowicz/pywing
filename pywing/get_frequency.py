from statsmodels.tsa.stattools import acf

'''
This function uses the statsmodels.tsa.stattools function acf or autocorrelation
function to find the most likely frequency for the data. What this function
does is look at all the lags, up to the number of lags specified by lag_corr,
and computes the correlation between the original point and the lag point.
The maximum absolute correlation is taken as a proxy for a frequency. For 
example, we may find a strong correlation between points that are 
7, 14, 21, etc. days apart if the sender is a weekly sender.
INPUTS:
x           = A vector of data
lag_corr    = The maximum number of lags to consider
OUTPUT:
freq+1      = The lag with the maximum correlation, which is a proxy for the
              frequency. A +1 is included for readability since Python is 
              0-indexed, and because the slicing done on the freq vector
              equates a 0-index with a 1-day lag.
'''
def get_frequency(x, lag_corr):
    freq = np.abs(acf(x, nlags=lag_corr, qstat=True)[0][1:]).argmax()
    return freq+1 #+1 b/c it is 0-indexed

