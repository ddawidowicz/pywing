import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.dates import YearLocator, MonthLocator, \
						WeekdayLocator, DateFormatter, MO
margin = 0.10 * (np.max(x) - np.min(x))

    #years    = YearLocator()   # every year
    months = MonthLocator()  # every month
    weeks = WeekdayLocator(byweekday=MO, interval=1)
    #yearsFmt = DateFormatter('%Y')
    #monthFmt = DateFormatter('%b')
    wkFmt = DateFormatter('%m-%d-%y')

    fig, ax = plt.subplots()
    ax.plot_date(dates, x, '-', lw=3, label='Volume')
    for i in xrange(len(xmin_list)):
        print yint_list[i], xmin_list[i], xmax_list[i]
        ax.axhline(y=yint_list[i], xmin=xmin_list[i], xmax=xmax_list[i], label='Mean', lw=4, color='r')

    # format the ticks
    #ax.xaxis.set_major_locator(months)
    #ax.xaxis.set_major_formatter(monthFmt)
    ax.xaxis.set_major_locator(weeks)
    ax.xaxis.set_major_formatter(wkFmt)
    ax.autoscale_view()

    # format the coords message box
    def get_vol(y,pos): return '{:0,d}'.format(int(y)) 
    #have to include pos even though not used b/c of FuncFormatter passing value and position
	#For stupid 2.6
		import locale
		locale.setlocale(locale.LC_ALL, 'en_US')
		def get_comma(y,pos): return locale.format("%d", int(y), grouping=True)
		ax.yaxis.set_major_formatter(tkr.FuncFormatter(get_comma))
    #ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    #ax.fmt_ydata = get_vol
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(get_vol))
    ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.yaxis.grid(True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.set_title("%s\nMessage Volume by Week" % my_domain)
    #ax.grid(True)

    fig.autofmt_xdate()
    ax.legend(loc=0)
    plt.show()
