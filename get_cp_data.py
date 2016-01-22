
'''
This function performs change point analysis and returns a list of information
related to the change points found.
INPUTS:
dates           = The dates for the data (vector)
x               = The data vector, e.g. volume data
lag_max         = The maximum lag to use during the differencing step prior
                  to performing change point analysis. The differencing is
                  repeated until autocorrelation is sufficiently removed
                  and may actually difference the data by as much as
                  1.5 * lag_max. For example, if 14 is used for lag_max,
                  the data may be differenced by 21 days.
lag_corr        = The maximum number of lags to use while computing the 
                  independce of the elements in the vector during differencing
sig_level       = The signficance level for the test of independence to use
                  while differencing to determine if autocorrelation has been
                  sufficiently removed. A common value is 0.05.
CI              = The confidence on the test for significance of the change
                  point analysis. It turns out that the best results use a 
                  very generous value, e.g. ~0.20, for this confidence for 
                  initial discovery of change points and then relies on 
                  filtering of these candidate points.
N               = The number of bootstrap samples for determining the CI of 
                  a change point.
prune_period    = The minimum number of days between change points. The
                  process of change point analysis may put two change points
                  only a day apart which clouds this issue of which is the
                  best point to examine. This threshold forces change points
                  to be a certain number of days apart, e.g. 7 days.
OUTPUTS:
cp_pruned       = The list of change points. Each element of the list is
                  another list with the form 
                  [idx, confidence, from_mean, to_mean, level]
y_mean          = A vector of data that contains the mean for that subsection,
                  broken by the change points, for each value of x. This allows
                  for plotting the mean horizontal line for each subsection
                  defined by the change points.
y_cp            = A point on the end of the mean to mark the change point 
                  itself. The rest of the values in the vector are None.
cp_list         = A friendly-formatted version of cp_pruned for user display.
'''
def get_cp_data(dates, x, lag_max, lag_corr, sig_level, CI, N, prune_period):
    # --------------------- Remove Autocorrelation ----------------------- #
    [x_diff,x_box,lag_used] = remove_autocorr(x,lag_max,lag_corr,sig_level)
    
    # ------------------ Run Change point Analysis ----------------------- #
    #The cp_pruned have form [idx, confidence, from_mean, to_mean, level]
    cp = change_points(x, x_diff, lag_used, CI, N)
    cp_pruned = prune_cp(cp, prune_period, 1) #the 1 is index of the confidence
    cp_pruned = add_means(x, cp_pruned, lag_used)
    cp_pruned.sort(key=lambda x: x[0])
    
    # ------------------- Gather Change point Data ----------------------- #
    y_mean = [np.mean(x) for i in xrange(len(x))]
    y_cp = [None for i in xrange(len(x))]
    ref_idx = 0
    for c in cp_pruned:
        idx = int(c[0])
        from_mean = c[2] #'%.2f' % c[2]
        y_mean[ref_idx:idx] = [from_mean for i in xrange(idx-ref_idx)]
        y_cp[(idx-1)] = from_mean #-1 b/c want to sit on from_mean not to_mean
        ref_idx = idx
    try:
        y_mean[ref_idx:] = cp_pruned[-1][3] #get the last mean value (to_mean)
    except:
        #again can format to string for 2 decimal places
        y_mean[ref_idx:] = [np.mean(x[ref_idx:]) for i in xrange(len(x[ref_idx:]))]
    
    
    # ------------------- Create Change point List ----------------------- #
    cp_list = map(lambda y: [get_date_obj(dates[int(y[0])]), \
                            ('%.2f' % y[2]), \
                            ('%.2f' % y[3]), \
                            ('%.2f' % (100.0*y[1]))], \
                            cp_pruned)

    return [cp_pruned, y_mean, y_cp, cp_list]    

