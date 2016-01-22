import numpy as np
from statsmodels.tsa.stattools import acf


'''
This function removes autocorrelation from data by repeated differencing. This is
the only function that needs to be called.
INPUTS:
x           = A vector of data
lag_max     = The maximum number of lags to consider for differencing
lag_corr    = The number of lags to use in the test for independence
sig_level   = A float that determines the cutoff for whether the vector of data
              has reached an acceptable state of independence. In general
              anything below 0.05 signifies significant autocorrelation.
 
OUTPUTS:
x_diff      = The vector of differenced data
x_box       = The p-value from the Box.test above sig_level indicates independence 
offset      = The cummulative number of lags that were used in the differencing.
              This number should be added to any index to map back to the oringinal
              data vector.

NOTE:
The only function that needs to be called is remove_autocorr(), the other two are
for internal use only.
'''
def remove_autocorr(x, lag_max, lag_corr, sig_level):
    #Check if data is already significantly independent
    x_box = acf(x, nlags=lag_corr, qstat=True)[2][-1]
    if x_box >= sig_level:
        lag_used = 0
        return [x, x_box, lag_used]

    #check with two differencing, using lag_max as the maximum lag for
    #the first pass and lag_max/2 as the maximum lag for the second pass.
    #Will stop after the first differencing if needed.
    [x_diff, x_box, lag_used] = diff_twice(x, lag_max, lag_corr, sig_level)

    #if two differencings (or one if stopped early) are enough, return results
    if x_box >= sig_level:
        return [x_diff, x_box, lag_used]

    #Check with two more differencing, using 2*lag_max as the maximum lag for
    #the first pass and lag_max as the maximum lag for the second pass.
    #Will stop after the first differencing if needed.
    [x_diff, x_box, lag_used] = diff_twice(x, (2*lag_max), lag_corr, sig_level)

    if x_box >= sig_level:
        return [x_diff, x_box, lag_used]
    else:
        [x_diff, x_box, lag_used] = diff_repeat(x, lag_max, lag_corr, sig_level)

    return [x_diff, x_box, lag_used]


'''
This function runs two differences, stopping after one difference if possible.
This function is used internally.
INPUTS:
x           = The original data vector
lag_max     = The maximum lag considered for differencing
lag_corr    = The number of lags to consider for the Box test
sig_level   = The threshold used to reject the null hypothesis that the data is independent
OUTPUTS:
x_diff      = The vector of differenced data
x_box       = The p-value from the Box.test above sig_level indicates independence 
overall_lag = The cummulative number of lags that were used in the differencing.
              This number should be added to any index to map back to the oringinal
              data vector.
'''
def diff_twice(x, lag_max, lag_corr, sig_level):
    [x_diff1, x_box1, lag_used1] = diff_with_best_lag(x, lag_max, lag_corr, sig_level) 
    
    if x_box1 >= sig_level:
        return [x_diff1, x_box1, lag_used1]
    else:
        [x_diff2, x_box2, lag_used2] = diff_with_best_lag(x, int(lag_max/2), \
                                                        lag_corr, sig_level) 
    overall_lag = lag_used1 + lag_used2
    return [x_diff2, x_box2, overall_lag]


'''
This function repeats the differencing until sig_level/10.0 is reached. I chose to
decrease the threshold since this function is only called after other attempts have
been made and we are at risk of loosing too much data.
This function is used internally.
INPUTS:
x           = The original data vector
lag_max     = The maximum lag considered for differencing
lag_corr    = The number of lags to consider for the Box test
sig_level   = The threshold used to reject the null hypothesis that the data is independent
OUTPUTS:
x_diff      = The vector of differenced data
x_box       = The p-value from the Box.test above sig_level indicates independence 
overall_lag = The cummulative number of lags that were used in the differencing.
              This number should be added to any index to map back to the oringinal
              data vector.
'''
def diff_repeat(x, lag_max, lag_corr, sig_level):
    overall_lag = 0
    x_box = 0.0
    x_copy = x[:] #used in case of failure
    try:
        while x_box < (sig_level/10.0):
            [x_diff, x_box, lag_used] = diff_with_best_lag(x, lag_max, lag_corr, sig_level) 
            overall_lag += lag_used
            x = x_diff
    except:
        #It is possible the above will fail if it runs out of points during differencing
        #This is a stop-gap in that case       
        [x_diff, x_box, overall_lag] = diff_twice(x_copy, (2*lag_max), lag_corr, sig_level)
    return [x_diff, x_box, overall_lag]


'''
This function looks for the best lag from (1,lag_max), i.e. the lag that removes 
the most autocorrelation.
This function is used internally.
INPUTS:
x           = The original data vector
lag_max     = The maximum lag considered for differencing
lag_corr    = The number of lags to consider for the Box test
sig_level   = The threshold used to reject the null hypothesis that the data is independent
OUTPUTS:
x_diff      = The vector of differenced data
x_box       = The p-value from the Box.test above sig_level indicates independence 
overall_lag = The cummulative number of lags that were used in the differencing.
              This number should be added to any index to map back to the oringinal
              data vector.
'''
def diff_with_best_lag(x, lag_max, lag_corr, sig_level=0.05):
    lag_used = 0
    ref = 0.0
    for lag in xrange(1,(lag_max+1)):
        x_diff = x[lag:]-x[:-lag]
        #In the Box.test the lag_max serves the purpose not of limiting the 
        #choices but it is the number of lags to consider in the statistic.
        x_box = acf(x_diff, nlags=lag_corr, qstat=True)[2][-1]
        if np.abs(x_box) > ref:
            lag_used = lag
            ref = x_box
    x_diff = x[lag_used:] - x[:-lag_used]
    return [x_diff, x_box, lag_used]

