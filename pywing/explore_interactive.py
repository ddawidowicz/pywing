import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_help():
    print 'Inputs:'
    print '--------------------------------------------------------------------'
    print 'df = data frame (use df = get_interactive_df())'
    print 'rollup = column name(s) on which to summarize data (accepts list)'
    print "sort_key = column(s) on which to sort summary data (accepts list " \
            "and use '' for an unsorted summary"
    print 'disp_cols = which columni(s) to display in the summary ' \
            '(accepts list and use [] for all columns'
    print 'cond = 2-element tuple or list where the first element is the ' \
            'column of interest and the second element is the specific value ' \
            'to select'
    print 'xvar = the name of the column used for the x-variable'
    print 'yvar = the name of the column used for the y-variable'
    print '--------------------------------------------------------------------'
    print 'Available functions:'
    print 'df = get_interactive_df()'
    print 'df1 = get_rollup_sum(df, rollup, sort_key, disp_cols)'
    print 'df1 = get_rollup_mean(df, rollup, sort_key, disp_cols)'
    print 'plot_trend(df, cond, xvar, yvar)'
    print '--------------------------------------------------------------------'

def get_interactive_df():
    #[fulcrum_ip_domain, fulcrum_model_vars, 
    #fulcrum_delivery_metrics, fulcrum_subscriber_model_vars
    # ============================ Control Panel ============================= #
    infile = raw_input('Enter input file (include file extension):  ')
    sheet_idx = raw_input('Enter sheet index:  ')
    # ======================================================================== #
    df = get_data(infile, int(sheet_idx), new_cols=[], sort_col='day')
    return df


def get_data(infile, idx, new_cols=[], sort_col='day'):
    print 'Retrieving data from %s' % infile
    print '(May take a moment)'
    data = pd.ExcelFile(infile)
    sheet = data.sheet_names[idx]

    print 'Using data set %s' % sheet
    df = data.parse(sheet)    
    df.sort(sort_col, inplace=True)
    for c in new_cols:
        df[c[0]] = df[c[1]] / (1.0 * df[c[2]])

    return df


def get_rollup_sum(df, rollup, sort_key, cols):
    if sort_key == '':
        summary = df.groupby(rollup).sum().reset_index()
    else:
        summary = df.groupby(rollup).sum().sort(sort_key).reset_index()
    if not cols:
        print 'Using all columns'
        cols = list(df.columns)
    
    if rollup not in cols:
        cols.extend(rollup)
    print 'cols = ', cols
    return summary[cols]


def get_rollup_mean(df, rollup, sort_key, cols):
    keys = df[rollup].unique()
    summary = df.groupby(rollup).mean().sort(sort_key).reset_index()
    if not cols:
        print 'Using all columns'
        cols = summary.columns
    
    return summary[cols]


def plot_trend(df, cond, xvar, yvar):
    if cond:
        x = df[df[cond[0]]==cond[1]][xvar]
        y = df[df[cond[0]]==cond[1]][yvar]
    else:
        x = df[xvar]
        y = df[yvar]
    if xvar == 'day':
        x -= 20140000
        labels = [str(i)[0] + '/' + str(i)[1:] for i in x]
    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title('Delivery metrics\n%s' % cond[1])
    #plt.show()
    return [plt, fig, ax]


if __name__ == '__main__':
    #[fulcrum_ip_domain, fulcrum_model_vars, 
    #fulcrum_delivery_metrics, fulcrum_subscriber_model_vars
    # ============================ Control Panel ============================= #
    infile = 'fulcrum_tables.xlsx'
    sheet_idx = 1 
    sort_key = ''
    rollup = ['day', 'domain']
    disp_cols = ['gmail_total','gmail_inbox', 'gmail_inbox_rt']
    new_cols = [('gmail_inbox_rt', 'gmail_inbox', 'gmail_total')]
    # ======================================================================== #

    df = get_data(infile, sheet_idx, new_cols)
    df1 = get_rollup_sum(df, rollup, sort_key, disp_cols)
    [plt, fig, ax] = plot_trend(df1, ['domain', 'some_domain'], 'day', 'gmail_total')
    plt.show()
    

