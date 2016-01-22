import numpy as np
import pandas as pd
import ggplot
from ggplot import aes, geom_line, geom_point, geom_hline, \
        scale_x_date, scale_colour_manual, scale_y_continuous, \
        xlab, ylab, ggtitle, ylim, theme_seaborn, date_format
'''
'''


def plot_vol(dates, x, cp, my_domain):
    # -------------------- Prepare for Plotting -------------------------- #
    # Prepare DataFrame objects for graphing
    #Add a column for the label to show in the legend in the graph
    #Need to reshape it, from (124,) to (124,1) for exmple, so that it
    #will concatenate. This gives a df with [date, vol_data, 'Volume']
    v = ['Volume' for i in xrange(x.shape[0])]
    #df_domain = np.concatenate((x, v), axis=1)
    ndf_vol = np.transpose(np.array([dates, x, v]))
    df_vol = pd.DataFrame(ndf_vol, columns=['Date', 'Volume', 'Data'])

    #Create pre-allocated lists for plotting means and cp
    xmin_list = [0 for i in xrange(len(cp))] #hold lft pt of vol_mean
    xmax_list = [0 for i in xrange(len(cp))] #hold rt pt of vol_mean
    yint_list = [0 for i in xrange(len(cp))] #holds vol_means
    cp_date_list = [0 for i in xrange(len(cp))] #holds date for cp
    cp_value_list = [0 for i in xrange(len(cp))] #holds cp value

    ref_idx = 0 #used to keep track of vol_means
    #collect list data for plotting
    for i in xrange(len(cp)):
        cp_idx = cp[i][0]-1 #-1 b/c 1-indexed (includes cp itself)
        xmin_list[i] = dates[ref_idx].toordinal() #convert to match ggplot
        xmax_list[i] = dates[cp_idx].toordinal() #convert to match ggplot
        yint_list[i] = cp[i][2] #use value from_mean for vol_mean
        cp_date_list[i] = dates[cp_idx] #date of cp
        #cp_value_list[i] = x[cp_idx] #value of cp
        cp_value_list[i] = cp[i][2]
        ref_idx = cp_idx + 1 #+1 b/c moving to next point
    
    #Reform lists into a data frame and attach to df_domains. The first two 
    #lists can be created together since they are both numeric, but if I try
    #to create all three together all types will be downgraded to strings.
    #np.concatenate avoids this conversion. The transpose is needed to take
    #an item from each to form a single row.
    cp_lbl = ['Change Point' for i in xrange(len(yint_list))]

    #Need to create a dummy entry to put 'Volume Mean' into legend
    cp_date_list.append(dates[0])
    yint_list.append(x[0])
    cp_lbl.append('Volume Mean')
    ndf_cp = np.transpose( np.array([cp_date_list, yint_list, cp_lbl]) )
    yint_list.pop(-1)
    cp_date_list.pop(-1)
    df_cp = pd.DataFrame(ndf_cp, columns=['Date', 'Volume', 'Data']) 

    df_plot = pd.concat( (df_vol, df_cp), axis=0)


    #Need to create a dummy entry to put 'Volume Mean' into legend
    #dummy = np.array([dates[0], x[0], 'Volume Mean']).reshape(1,-1)
    #df_cp = np.concatenate( (df_cp, dummy), axis=0) #add to bottom df_cp
    #df_domain = np.concatenate( (df_domain, df_cp), axis=0 ) #add df_domains

    #convert final array into a pd.DataFrame for printing and plotting
    #df_domain = pd.DataFrame(df_domain, columns=['Date','Volume','Data']) 
    #df_domain.to_html(open('out.html','w'))
    #os.system('sudo cp out.html /usr/local/www/analytics/rwing')

    margin = 0.10 * (np.max(x) - np.min(x))
    p = ggplot.ggplot(aes(x='Date', y='Volume', color='Data'), data=df_plot) + \
            ggplot.geom_line(color='blue',size=2) + \
            ggplot.geom_point(x=xmax_list, y=cp_value_list, color='black', \
                        shape='D', size=50) + \
            ggplot.geom_hline(xmin=xmin_list, \
                        xmax=xmax_list, \
                        yintercept=yint_list, color="red", size=3) + \
            ggplot.scale_x_date(labels = date_format("%Y-%m-%d"), breaks="1 week") + \
            ggplot.scale_colour_manual(values = ["black", "blue", "red"]) + \
            ggplot.scale_y_continuous(labels='comma') + \
            ggplot.ylim(low=np.min(x)-margin/4.0, high=np.max(x)+margin) + \
            ggplot.xlab("Week (Marked on Mondays)") + \
            ggplot.ylab("Message Vol") + \
            ggplot.ggtitle("%s\nMessage Volume by Week" % my_domain) + \
            ggplot.theme_seaborn()

    return p
