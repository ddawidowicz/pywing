from statsmodels.nonparametric.smoothers_lowess import lowess

'''
This function performs lowess smoothing over the data to produce a trend line.
Basically this is like a moving average, but a little smarter. It computes
local linear regression models around each point, where local is defined
as some percentage of the data, and uses the model to predict where the point
"should" be to follow the trend. The resulting vectors is a smoothed version
of the original. The smooth_f parameter determines the extent of the smoothing.
INPUTS:
x           = The original data vector, e.g. daily volume
smooth_f    = The percentage of points to use for smoothing. This determines
              the local neighborhood around each point for the linear
              regression model creation. I used 0.2, but also thought it
              might be natural to use freq/len(x)
OUTPUT:
trend       = The smoothed data vector.
'''
def get_trend(x, smooth_f):
    # ---------------------- Run Trend Analysis ----------------------- #
    obs_idx = range(len(x))
    trend = lowess(x, obs_idx, frac=smooth_f, it=10, return_sorted=False)
    return trend

