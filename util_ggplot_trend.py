import ggplot
import pandas as pd
from ggplot import aes, geom_line, ggtitle, scale_x_date, \
            ylab, theme_seaborn, date_format, facet_wrap

'''
'''


def plot_trend_season(dates, ndf_domain, x, x_trend, season, my_domain):
    # ---------------------- Prepare Data Frame ----------------------- #
    df_domain = pd.DataFrame(ndf_domain, columns=['Date', 'Volume'])
    df_domain['Date'] = dates

    x_lbl = ['Observed Volume' for i in xrange(len(x))]
    xt_lbl = ['Overall Trend' for i in xrange(len(x_trend))]
    xs_lbl = ['Repeat Sending Trend' for i in xrange(len(season))]
    col3 = pd.DataFrame(x_lbl+xt_lbl+xs_lbl)

    df_plot = pd.concat( (df_domain, col3), axis=1)
    df_plot.columns = ['Date', 'Volume', 'Data']
    
    
    # ---------------------- Plot Decomposition ----------------------- #
    p = ggplot.ggplot(aes(x='Date', y='Volume', color='Data'), data=df_plot) + \
        ggplot.geom_line(color='blue', size=2) + \
        ggplot.scale_x_date(labels = date_format("%Y-%m-%d"), breaks="1 week") + \
        ggplot.xlab("Week (Marked on Mondays)") + \
        ggplot.ylab("Message Vol") + \
        ggplot.ggtitle("%s Message Volume by Week" % my_domain) + \
        ggplot.facet_grid('Data', scales='free_y') + \
        ggplot.theme_seaborn()

    return p
