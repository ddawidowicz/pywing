'''
This module contains a set of functions for calculating change-points within
a set of data. The only function that needs to be called is change_points().
This will take in a vector of data and output a list of change_points (if any)
and the confidence level for that change-point, which corresponds to the 
Mann Whitney (Wilcox) U test. This test is similar in concept to a t-test,
but unlike the t-test, it does not assume normally distributed data.
Also note that a bootstrap confidence interval is calculated, but not used
in the get_candidate_pts() function. This looks at the confidence interval
as the number of random shufflings beat / number of shufflings.
This website was helpful, http://www.variation.com/cpa/tech/changepoint.html
USUAGE:
>> from util.change_point import change_points
>> change_pts = change_points(data, CI=0.95,N=1000,ranked=False,verbose=False)
INPUTS:
data    = A vector of time-series data (original, not necessarily independent)
x       = A vector of time-series data (potential dependence removed). This
          could also be the same vector as data if data is already independent.
lag_used= If data and x are different, i.e. x has been differenced in some way,
          lag_used is the amount of lag that was used in the differencing.
CI      = The confidence interval to use for the change-point detection
N       = The number of bootstrap samples to use for change-point detection
ranked  = A boolean value that determines if ranks are used instead of the
          values of the data vector. This can help compensate in the presence
          of outliers.
verbose = A boolean value that will print intermittent results and progress
OUTPUT:
final_pts = A list of lists. Each inner list contains the change-point detected
            as a first element and the percentage of bootstrap samples it beat.
            The first item in this list is the level 1, i.e. most impactful 
            change-point, the next item is one of the level 2 results (lower
            half) and the first entry greater than the level 1 point is another
            level 2 point for the upper half.
'''

import pudb
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu
from itertools import chain

'''
This is the main function to call - see description above. 
'''
def change_points(data, x, lag_used=0, CI=0.95, N=1000, ranked=False, verbose=False):
    if ranked:
        x = get_rank(x)
    ref_idx = 0 #used for recursion in get_candidate_pts
    depth = 1 #initial change-point has depth=1
    pts = [] #holds change_points for each recursion
    change_pts = get_candidate_pts(x, pts, ref_idx, CI, N, depth)
    change_pts = get_cp_signif(x, change_pts) #assumes independence (use x not data)
    change_pts = add_means(data, change_pts, lag_used)
    return change_pts


'''
This function is a recursive function that finds a change point, splits the
data into two parts at that point, and recurses on each of the two smaller
subsets. This function is only called if a change point is detected.
INPUTS:
data        = The original data vector
pts         = Comes in as an empty list, but is passed recursively and holds
              the change points found
ref_idx     = Comes in as 0, but takes the value of the change-point found 
              and is used as an offset for the upper half of any split to 
              keep all change-points in reference to the original data vector.
CI          = The confidence interval used for change-point detection
N           = The number of bootstrap samples to use when detecting a change-pt
OUTPUT:
pts         = The list of change-points found
'''
def get_candidate_pts(data, pts, ref_idx, CI, N, depth):
    if len(data) < 7: #arbitrary choice - chose a week as smallest term
        return pts

    [_, Sdiff] = get_cumsum_data(data)
    [change, conf_level] = detect_change(data, Sdiff, CI, N)
    if change == 0:
        return pts
    else:
        idx = get_change_pt(data)

    pts.append([(idx + ref_idx), conf_level, depth])
    pts = get_candidate_pts(data[:idx], pts, ref_idx, CI, N, depth+1)
    pts = get_candidate_pts(data[idx+1:], pts, (idx+ref_idx), CI, N, depth+1) #+1 don't incld pt

    return pts


'''
This function looks at each point in the data vector and finds the point with
the smallest mse, which corresponds to the most likely change-point. This
function is only called if a change-point is detected.
INPUT:
data        = The data vector being examined. Could be the whole data vector or
              a subset from get_candidate_pts()
OUTPUT
compare[0]  = The index of the change-point found
'''
def get_change_pt(data):
    compare = [-1,np.inf] #[index, value]
    for m in xrange(1, len(data)):
        first = data[:m]
        second = data[m:]
        x1 = np.mean(first)
        x2 = np.mean(second)
        s1 = np.sum( np.power(first-x1, 2) )
        s2 = np.sum( np.power(second-x2, 2) )
        mse = s1 + s2
        if mse < compare[1]:
            compare[0] = m
            compare[1] = mse
    return compare[0]


'''
This function detects whether or not there is a change point in the time series
data.
INPUTS:
data        = The data vector
Sdiff       = The change-point value, retrieved from get_cumsum_data(y) above
CI          = The confidence interval to use for detecting change-points,
              e.g. 0.95 for a 95% confidence interval
N           = The number of bootstrap samples to use
OUTPUTS:
change      = Boolean value, 1=change-point(s) detected, 0=no change-points
conf        = Confidence value on whether there is a change-point. This is the
              percent of bootstrap samples that were beaten.
'''
def detect_change(data, Sdiff, CI, N):
    data_cp = np.copy(data)
    count = 0
    Sboot = np.zeros(N)
    for i in xrange(N):
        np.random.shuffle(data_cp)
        [_, Sboot[i]] = get_cumsum_data(data_cp)
    count = np.sum(Sboot < Sdiff)
    conf = count / (1.0 * N)
    if conf > CI:
        change = 1
    else:
        change = 0
    return [change, conf]
        

'''
This function takes in a data vector and returns a vector of the data point
ranks rather than values. This is useful if there are outliers in the data.
INPUT:
data    = The original data vector
OUTPUT:
ranked  = A vector of the ranks of the values
'''
def get_rank(data):
    ranked = stats.rankdata(data)
    return ranked


'''
This function computes the cumsums of the differences between each data
point and the overall mean and computes the difference between the max
and min cumsum for comparison with bootstrap samples.
INPUT:
y       = A vector of time-series data
OUTPUTS:
ycs     = The cumsum vector of each data point and the mean
Sdiff   = The main change-point variable, max(ycs) - min(ycs)
'''
def get_cumsum_data(y):
    yd = y - np.mean(y)
    ycs = np.cumsum(yd)
    ycs = np.insert(ycs, 0, 0) #always include a zero at the beginning
    Sdiff = np.max(ycs) - np.min(ycs) #the main metric for change-point analysis
    return [ycs, Sdiff]


'''
This function uses the change points found on the independent data set (x in the
main function above) and computes the means from the original data set (data in 
the main function above).
INPUTS:
data        = A vector of time-series data (original, not necessarily independent)
change_pts  = A list of change points found [idx, confidence, level]
lag_used    = If the data set had to be differenced to account for autocorrelation
              this value is the amount of differencing that was used and serves
              as a way to map the index of the change points (found on the
              differenced data) back to the original data vector
OUTPUT
complete_pts = A list of change points now of the form 
               [idx, confidence, from_mean, to_mean, level]
'''
def add_means(data, change_pts, lag_used):
    try:
        # ------------------------------ Add Means --------------------------- #
        x = np.array(change_pts) #put it into array for sorting
        x = x[x[:,0].argsort()] #sort by first column, i.e. the index
        x[:,0] += lag_used #adjust cp back to original data set (if lag_used != 0)
        complete_pts = x.tolist() # put it back into list for inserting
        ref_idx = 0
        for p in xrange(len(complete_pts)):
            idx1 = complete_pts[p][0]
            try:
                idx2 = complete_pts[p+1][0]
            except:
                idx2 = len(data)
            mean1 = np.mean(data[ref_idx:idx1]) #from
            mean2 = np.mean(data[idx1:idx2]) #to
            complete_pts[p].insert(2, mean1)
            complete_pts[p].insert(3, mean2) #insert after mean1
            ref_idx = idx1
        
        x = np.array(complete_pts) #put it into array for sorting
        x = x[x[:,1].argsort()] #sort by confidence
        complete_pts = x.tolist() # put it back into list 
        date_idx = map(int, [x[0] for x in complete_pts]) #convert index to int
        date_idx = map(int, [x[4] for x in complete_pts]) #convert level to int
        return complete_pts 
    except:
        return change_pts


'''
This function looks at all the candidate change points and prunes them so that we
don't have points right on top of each other. For example if the first change point
was at index=25, then the data is split into two separate data sets, data[:25] and
data[26:]. Looking at the first segment it is possible that the most meaningful
change point is at point 23 or something close to the original idx=25. This function
eliminates those confounding issues. The confidence level, Mann Whitney U test, 
is used to choose between neighboring points.
INPUTS:
cp              = The change points [idx, confidence, from_mean, to_mean, level]
prune_period    = The minimum number of days allowed between two change points
conf_idx        = The index into the confidence information
OUPUTS:
cp              = A pruned list of the same format as the incoming cp matrix
'''
def prune_cp(cp, prune_period, conf_idx):
    x = np.array(cp) #put it into array for sorting
    try:
        x = x[x[:,0].argsort()] #sort by first column, i.e. the index
    except:
        #1 dimensional
        x.sort()
    cp = x.tolist() # put it back into list for inserting
    for c, e in reversed(list(enumerate(cp[1:]))): #odd indexing, c+1, artifact of reverse
        if cp[c+1][0]-cp[c][0] < prune_period:
            if cp[c+1][conf_idx] > cp[c][conf_idx]:
                cp.pop(c)
            else:
                cp.pop(c+1)
    return cp


'''
This function uses the Mann-Whitney (Wilcox) U test to determine if two samples are
from different distributions. It is analogous to the Student t-test, but does not
assume normally distributed data as the t-test does. When the cp matrix comes in, 
it has a confidence value in place already, but this was based on the bootstrap
method described at the top of this file, and is subject to the length of the vector.
INPUTS:
x       = The data vector with the autocorrelation removed.
cp      = The matrix of change points, [idx, confidence, from_mean, to_mean, level]
OUPUTS:
cp      = The same matrix of change points, but with the confidence values replaced
          with the Mann-Whitney U test results. I use 1-p.value to give the probability
          of rejecting the null rather than the probability of the null.        
'''
def get_cp_signif(x, cp):
    for c in cp:
        chng_pt = c[0]
        prob = 1-mannwhitneyu(x[:chng_pt], x[(chng_pt+1):])[1]
        c[1] = prob
    return cp



'''
This function prints the results of the change point analysis to stdout.
INPUTS:
change_pts  = The matrix of change points, [idx, confidence, from_mean, to_mean, level]
dates       = A vector of the dates associated with the data analyzed
msg         = A header above the table of results
OUTPUT:
A table displayed to stdout
'''
def print_results(change_pts, dates, msg=''):
    try:
        x = np.array(change_pts) #put it into array for sorting
        x = x[x[:,0].argsort()] #sort by first column, i.e. the index
        cp = x.tolist() # put it back into list for inserting
        print msg
        print '%10s\t%10s\t%10s\t%10s\t%10s\t%10s' % \
            ('Index(1-idx)','Date', 'Confidence', 'From', 'To', 'Level')
        for i in xrange(len(cp)):
            d_idx = cp[i][0]
            date = str(dates[d_idx])[:4] + '-' + str(dates[d_idx])[4:6] + \
                        '-' + str(dates[d_idx])[6:]
            print '%10d\t%10s\t%10.2f%%\t%10.2f\t%10.2f\t%10d' % \
                (d_idx, date, 100.0*cp[i][1], cp[i][2], cp[i][3], cp[i][4])
    except:
        print msg
        print 'No change-points detected'


'''
This function writes the results of the change point analysis to two different files.
INPUTS:
outlog1         = This is the file name (with path, if necessary, and ext.) of the
                  readable results file. The results written here are easy to read
                  but not intended to be used further programmatically.
outlog2         = This is the file name (with path, if necessary, and ext.) of a
                  file (usually .csv) that can be used for further analysis. It prints
                  each change point on its own line in the format:
                  idx,date(e.g. 20140523),confidence,from_mean,to_mean,level
change_pts      = A matrix of the results were each row represents one cp and is like
                  [idx, confidence, from_mean, to_mean, level].
dates           = A vector of dates matching the length of the original data vector
msg             = A message used above the results table in outlog1 only
OUTPUTS:
outlog1         = See above - returned for further writing or closing
outlog2         = See above - returned for further writing or closing
'''
def write_results(outlog1, outlog2, change_pts, dates, msg):
    try:
        x = np.array(change_pts) #put it into array for sorting
        try:
            x = x[x[:,0].argsort()] #sort by first column, i.e. the index
        except:
            #1 dimensional
            x.sort()
        cp = x.tolist() # put it back into list for inserting
        outlog1.write('\n%s\n' % msg)
        outlog1.write('%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n' % \
            ('Index(1-idx)','Date', 'Confidence', 'From', 'To', 'Level'))
        
        outlog2.write('%s,%s,%s,%s,%s,%s\n' % \
                    ('Index(1-idx)','Date', 'Confidence', 'From', 'To', 'Level'))
        for i in xrange(len(change_pts)):
            d_idx = cp[i][0]
            date = str(dates[d_idx])[:4] + '-' + str(dates[d_idx])[4:6] + \
                        '-' + str(dates[d_idx])[6:]
            outlog1.write('%10d\t\t%10s\t%10.2f%%\t%10.2f\t%10.2f\t%10d\n' % \
                (d_idx, date, 100.0*cp[i][1], cp[i][2], cp[i][3], cp[i][4]))
            
            outlog2.write('%d,%d,%.4f,%.2f,%.2f,%d\n' % \
                        (d_idx,dates[d_idx], cp[i][1], cp[i][2], cp[i][3], cp[i][4]))
        outlog1.write('\n')
    except:
        outlog1.write('No change-points detected\n\n')
    
    return [outlog1, outlog2]



'''
Creates a plot of the cummulative sum of the differences beteween each data
point and the mean of the data vector. NOT USED. MAY WANT TO EDIT TO SAVE PLOT.
INPUTS:
x   = A vector of indices for each cumsum
y   = a vector of cumsums of the differences between each point and the mean.
OUTPUT:
displays a plot, may want to edit to save plot instead.
'''
def plot_cumsum(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel('X-variable')
    ax.set_ylabel('CUMSUM')
    ax.set_title('CUMSUM Chart')
    plt.show()


'''
A test function that runs the different functions with a small data set.
see, http://www.variation.com/cpa/tech/changepoint.html.
EXPECTED OUTPUT:
point 11 with 100% confidence
point 5 with 0-20% confidence (possibly).
'''
if __name__ == '__main__':
    start = time.time()
    data = np.array([10.7,13.0,11.4,11.5,12.5,14.1,14.8,14.1,12.6,16.0,11.7,10.6, \
            10.0,11.4,7.9,9.5,8.0,11.8,10.5,11.2,9.2,10.1,10.4,10.5])
    change_pts = change_points(data)
    print_results(change_pts)
    end = time.time()
    print '\nElapsed time = %.4f seconds' % (end-start)


'''
DEPRECIATED - Use get_cp_signif() instead. This one views CI as the number of
bookstrap samples that are beaten, but this is subject to the length of the
sequence. The result is that change points found on lower levels have shorter
sequences and therefore get a higher significance rating than the high-leveled
change points, which is not what we want.

This function takes in a list of change points (only the points) and returns
the confidence level of each point along with the level and means. I used this
originally in conjunction with R. An R script gave me the change points and I
used this function to get those points into the same structure as the change
points generated from this module itself so I could use the tabular outputting
functions.
INPUTS:
data        = A vector of time-series data
cp          = A list of change points
N           = The number of bootstrap samples to use for change-point detection
OUTPUT:
change_pts  = A list of lists. Each inner list contains the change-point detected
            as a first element and the percentage of bootstrap samples it beat.
            The first item in this list is the level 1, i.e. most impactful 
            change-point, the next item is one of the level 2 results (lower
            half) and the first entry greater than the level 1 point is another
            level 2 point for the upper half.
'''
def get_cp_ci(data, cp, N):
    [_, Sdiff] = get_cumsum_data(data)
    cp_levels = [[] for i in xrange(len(cp))]
    start_idx = 0
    for c in xrange(len(cp)):
        try:
            end_idx = cp[c+1] #fails on last iteration b/c cp[c+1]
        except:
            end_idx = len(data) #to end of data set
        
        d = data[start_idx:end_idx]
        [_, conf_level] = detect_change(d, Sdiff, 0.0, N) #CI not used here
        cp_levels[c] = [cp[c], conf_level]
        start_idx = cp[c]
    change_pts = add_means_levels(data, cp_levels)
    return change_pts
