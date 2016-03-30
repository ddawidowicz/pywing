import pudb
import numpy as np
import locale
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from statsmodels.tsa.stattools import acf
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.dates import YearLocator, MonthLocator, WeekdayLocator, \
                            DateFormatter, MO
locale.setlocale(locale.LC_ALL, 'en_US')


'''
This function peforms an additive time series decomposition.
In classic time series decomposition we imagine the data vector Y to be 
composed of a trend element (T), a seasonal component (S), and 
random noise (R). In an additive model we can define the time series as
Y = T + S + R, where
T = some sort of moving average or better a weighted local regression
    model (Lowess or loess)
S = first find the trend, then Y - T = S + R. However, the season is a repeated
    portion of that quantity so we can find it by averaging all the points
    of S + R over the frequency. For example, if our frequency is 12, then
    we average points 1, 13, 25, ... then points 2, 14, 26, ..., etc.
R = Y - T - S
A very good reference is found at 
http://www.stats.ox.ac.uk/~burke/Autocorrelation/Decomposition%20and%20Smoothing.pdf
'''
def get_decompose(x):
    # ======================================================================== #
    #                           Control Panel                                  #
    # ======================================================================== #
    lag_corr = 14 #number of lags to consider in finding the frequency
    smooth_f = 0.20 #The fraction of points to use for smoothing 
    # ======================================================================== #

    freq = get_frequency(x, lag_corr)
    trend = get_trend(x, smooth_f)
    season = get_season(x, trend, freq)
    noise = get_noise(x, trend, season)

    return [trend, season, noise]


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


'''
This function uses the trend data found in get_trend() for time series 
decomposition resulting in a trend, season, and noise component to the
time series. The seasonal component, found by subtracting the trend from
the oringal time series and leveraging the frequency, extracts the underlying
repeat sending patterns.
INPUTS
x       = Original data vector (time series)
trend   = The trend vector found from get_trend()
freq    = The frequency of the time series found with get_frequency()
OUTPUT
season  = The vector representing the seasonal component of the time series
'''
def get_season(x, trend, freq):
    # -------------------- Run Seasonal Analysis ---------------------- #
    seas_err = x - trend
    season_means = np.zeros(freq)
    for i in xrange(freq):
        l = np.array(seas_err[i::freq]) #get every kth entry, k=freq
        season_means[i] = np.mean(l) #find the mean of this list
    season = np.tile(season_means, int(len(x)/(1.0*freq)) ) #may be short
    len_diff = len(x) - len(season) #find the difference
    season = np.concatenate((season,season_means[:len_diff])) #add for len
    return season


'''
The random noise of a time series (in an additive model) is found by
R = Y - T - S, where R is the random component, Y is the original data
vector, T is the trend vector, and S is the seasonal component.
'''
def get_noise(x, trend, season):
    noise = x - trend - season
    return noise 


'''
This function plots a time series decomposition plot containing the original
data, the trend data, and the seasonal data. It formats the dates to be
every week starting on Monday.
INPUTS:
dates   = A list of datetime objects
x       = The original data vector (numpy)
xt      = The trend vector (numpy)
xs      = The seasonal vector (numpy)
titles  = A list of three titles for the original, trend, and season plots
x_lbl   = The overall x-axis label, e.g. Week (Marked on Monday)
y_lbls  = A list of three y-axis labels for each plot
OUTPUT:
plt     = A matplotlib.pyplot object - you will need to call plt.show()
          or plt.savefig('foo.png', bbox_inches='tight')
'''
def plot_decomp(dates, x, xt, xs, titles, x_lbl, y_lbls):
    # ---------------------- Plot Decomposition ----------------------- #
    years    = YearLocator()   # every year
    #margin = 0.10 * (np.max(x) - np.min(x))
    months = MonthLocator()  # every month
    #weeks = WeekdayLocator(byweekday=MO, interval=1)
    yearsFmt = DateFormatter('%Y')
    monthFmt = DateFormatter('%b')
    #wkFmt = DateFormatter('%m-%d-%y')

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.subplots_adjust(hspace=0.3)
    ax1.plot_date(dates, x, 'b-', lw=3)
    ax2.plot_date(dates, xt, 'b-', lw=3)
    ax3.plot_date(dates, xs, 'b-', lw=3)

    # format the ticks
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(yearsFmt)
    #ax1.xaxis.set_major_locator(weeks)
    #ax1.xaxis.set_major_formatter(wkFmt)
    ax1.autoscale_view()
    #ax2.xaxis.set_major_locator(weeks)
    #ax2.xaxis.set_major_formatter(wkFmt)
    ax2.xaxis.set_major_locator(years)
    ax2.xaxis.set_major_formatter(yearsFmt)
    ax2.autoscale_view()
    #ax3.xaxis.set_major_locator(weeks)
    #ax3.xaxis.set_major_formatter(wkFmt)
    ax3.xaxis.set_major_locator(years)
    ax3.xaxis.set_major_formatter(yearsFmt)
    ax3.autoscale_view()

    # format the coords message box
    def get_comma(y,pos): return locale.format("%d", int(y), grouping=True)
    #ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    ax1.yaxis.set_major_formatter(tkr.FuncFormatter(get_comma))
    ax2.yaxis.set_major_formatter(tkr.FuncFormatter(get_comma))
    ax3.yaxis.set_major_formatter(tkr.FuncFormatter(get_comma))
    ax3.tick_params(axis='x', which='major', labelsize=10)
    ax1.xaxis.grid(True, 'major')
    ax1.xaxis.grid(True, 'minor')
    ax1.yaxis.grid(True)

    ax2.xaxis.grid(True, 'major')
    ax2.xaxis.grid(True, 'minor')
    ax2.yaxis.grid(True)
    
    ax3.xaxis.grid(True, 'major')
    ax3.xaxis.grid(True, 'minor')
    ax3.yaxis.grid(True)
    #ax.grid(True)
    #ax.ylim(low=np.min(x)-margin/4.0, high=np.max(x)+margin) + \
    
    ax3.set_xlabel(x_lbl)
    try:
        ax1.set_ylabel(y_lbls[0])
        ax2.set_ylabel(y_lbls[1])
        ax3.set_ylabel(y_lbls[2])
    except:
        pass
        #fig.text(0.02, 0.5, 'Message Volume', ha='center', va='center', \
        #                                            rotation='vertical')
    try:
        ax1.set_title(titles[0])
        ax2.set_title(titles[1])
        ax3.set_title(titles[2])
    except:
        pass

    fig.autofmt_xdate()

    #ax.legend(loc=0)
    #plt.show()

    return plt

if __name__ == '__main__':
    import datetime
    x = np.genfromtxt('ts_data.txt')
    [xt, xs, n] = get_decompose(x)
    print x
    print '-' * 40
    print xt
    print '-' * 40
    print xs
    print '-' * 40
    print n
    print '-' * 40

    numdays = len(x)
    base = datetime.datetime.today()
    dates = [base - datetime.timedelta(days=i) for i in xrange(numdays)]
    titles = ['title1', 'title2', 'title3']
    x_lbl = 'Week (Marked on Monday)'
    y_lbls = ['y1', 'y2', 'y3']
    myplt = plot_decomp(dates, x, xt, xs, titles, x_lbl, y_lbls)
    myplt.show()
