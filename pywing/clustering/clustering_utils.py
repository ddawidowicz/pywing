import os
import sys
import time
import MySQLdb
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from scipy import stats
sns.set(style="darkgrid")
pd.options.display.width = 250


# ========================== GENERAL PREPROCESSING =========================== #
'''
This function establishes a connection to the database.
INPUT:
db      = The name of the database
OUTPUT:
conn    = The connection object to be used to access the database
NOTE:
For now this is just using the localhost and the root user, but this will
change later. Eventually this function will read in the relevant information
from a configuration file.
'''
def get_db_connect(db):
    try:
        conn = MySQLdb.connect(host='localhost',
                                user='root',
                                passwd='',
                                db=db)
        return conn
    except Exception, e:
        "Unable to connect to the database"
        print '+' * 80
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, e, fname, exc_tb.tb_lineno)
        print '+' * 80



def get_data(db, query, get_cols=True):
    conn = get_db_connect(db)
    df_data = pd.read_sql(query, con=conn)
    conn.close()

    if get_cols:
        all_cols = df_data.columns
        #no_time_cols = all_cols[1:-22]
        utc_cols = all_cols[-22:-11]
        no_nan_cols_idx = range(1,25) + range(33,49)
        no_nan_cols = [all_cols[i] for i in no_nan_cols_idx]
        count_cols_idx = [1,2,6,7,8,10,11,13,14,18,19,20,24] + range(33,38) + range(39,49)
        count_cols = [all_cols[i] for i in count_cols_idx]
        ind_cols_idx = range(1, 7) + range(10, 25)
        ind_cols = [all_cols[i] for i in ind_cols_idx]
        collab_cols_idx = [7,8,9] + range(33, 49)
        collab_cols = [all_cols[i] for i in collab_cols_idx]
        cols_dict = {'all_cols':all_cols, 'no_nan_cols':no_nan_cols, \
                    'count_cols':count_cols, 'ind_cols':ind_cols, \
                    'collab_cols':collab_cols, 'utc_cols':utc_cols}
    else:
        cols_dict = None
        
    return [df_data, cols_dict]


def clean_data(df_data, cols_dict, utc_2007):
    # Get utc_cols
    utc_cols = cols_dict['utc_cols']

    # Get original df statistics
    print 'Original shape of df', df_data.shape

    # Get current time in utc
    now_utc = int(time.time())

    # Enter NaN if time > now (this is an error)
    before_utc_err = pd.isnull(df_data[utc_cols]).any(axis=1).nonzero()[0].shape[0]
    df_data.loc[:,utc_cols] = df_data[utc_cols].where(df_data[utc_cols] < now_utc, np.nan)

    # Make timestamps smaller and help to isolate errors
    df_data[utc_cols] -= utc_2007

    # Error if timestamps < 2007-10-01 (GitHub launch date)
    df_data.loc[:,utc_cols] = df_data[utc_cols].where(df_data[utc_cols] > 0, np.nan)
    after_utc_err = pd.isnull(df_data[utc_cols]).any(axis=1).nonzero()[0].shape[0]

    print 'Rows that had timestamp errors = %d' % (after_utc_err - before_utc_err)
    return df_data


def get_feature_sum_dist(df_data, cols_to_use, cutoff_params):
    lwr_bnd_prct = cutoff_params['lwr_bnd_prct']
    x_min = cutoff_params['x_min']
    x_max = cutoff_params['x_max']
    bin_width = cutoff_params['bin_width']
    cutoff = cutoff_params['cutoff']

    df_active = df_data[(df_data[cols_to_use] != 0).any(axis=1)]
    feature_sum = df_active[cols_to_use].sum(axis=1)
    prct = 100.0 - lwr_bnd_prct #look at top x% of users
    lwr_bnd = np.percentile(feature_sum, [prct])[0]
    prct_extreme = 100.0 * len(feature_sum[feature_sum > cutoff]) / float(df_data.shape[0])

    fig, ax = plt.subplots(figsize=(15,8))
    lbl1 = 'Cutoff = ' + '{0:,.0f}'.format(cutoff) + '\n(%.1f%% users above cutoff)' % prct_extreme
    ax.hist(feature_sum[feature_sum>lwr_bnd], \
            bins=np.arange(x_min,x_max+bin_width,bin_width), align='left', \
            label='Feature Sum\n(top %.1f%% of active users)' % lwr_bnd_prct)
    ax.axvline(x=cutoff+0.5*bin_width, lw=2.0, color='red', ls='--', label=lbl1)
    ax.set_title('Cutoff for Extreme Users', fontsize=16)
    ax.set_xlabel('Feature sum', fontsize=14)
    ax.set_ylabel('User count', fontsize=14)
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend([h1[1], h1[0]], [l1[1], l1[0]],loc=0, labelspacing=1.5, fontsize=14)

    return [cutoff, fig]


def separate_subgroups(df_data, cols_to_use, cutoff):
    # Remove users with no activity
    df_inactive = df_data[(df_data[cols_to_use] == 0).all(axis=1)]
    df_active = df_data[(df_data[cols_to_use] != 0).any(axis=1)]

    # Remove users with extreme activity (top ~2%)
    feature_sum = df_active[cols_to_use].sum(axis=1)
    df_extreme = df_active[feature_sum >= cutoff]
    df_active = df_active[feature_sum < cutoff]
    df_all = get_main_groups(df_inactive, df_active, df_extreme)

    print 'Cutoff = %d (avg count per feature = %.2f )' % (cutoff, cutoff / float(len(cols_to_use)))
    print 'Inactives (feature_sum = 0): ', df_inactive.shape
    print 'Actives (feature_sum < %d):' % cutoff, df_active.shape
    print 'Extremes (feature_sum >= %d):' % cutoff, df_extreme.shape
    print 'All users:', df_all.shape

    # Ensure no rows dropped accidentally
    new_rows = df_inactive.shape[0] + df_active.shape[0] + df_extreme.shape[0]
    assert (new_rows == df_data.shape[0]), \
            "You lost %d rows" % (df_data.shape[0] - new_rows)
    assert (df_all.shape[0] == df_data.shape[0]), \
            "You have a problem with df_all"

    return [df_all, df_inactive, df_active, df_extreme, feature_sum]


def get_main_groups(df_inactive, df_active, df_extreme):
    df_inactive.insert(df_inactive.shape[1], 'group', 0)
    df_active.insert(df_active.shape[1], 'group', 1)
    df_extreme.insert(df_extreme.shape[1], 'group', 2)
    df_all = pd.concat([df_inactive, df_active, df_extreme], axis=0)
    df_all.sort_values(by='user_id', inplace=True)
    return df_all


def eval_active_subgroups(feature_sum, norm_params, deg_free):
    h1 = norm_params['h1']
    mu1 = norm_params['mu1']
    sigma1 = norm_params['sigma1']
    h2 = norm_params['h2']
    mu2 = norm_params['mu2']
    sigma2 = norm_params['sigma2']

    [fig, x_intersect] = show_sub_groupings_t(feature_sum, \
                                            norm_params, deg_free)

    print 'Number of features =', deg_free + 1
    print 'Degrees of freedom =', deg_free
    print 'First group: mu = %.4f, 1 std = (%.4f, %.4f)' % \
                    (np.exp(mu1), np.exp(mu1 - sigma1), np.exp(mu1 + sigma1))
    print 'Second group: mu = %.4f, 1 std = (%.4f, %.4f)' % \
                    (np.exp(mu2), np.exp(mu2 - sigma2), np.exp(mu2 + sigma2))
    print 'X-value of equal probability (log, orig) = %.4f, %.4f' % \
                    (x_intersect, np.exp(x_intersect))
    print 'num std from mu2 = ', (mu2 - x_intersect) / sigma2
    print 'value on the other side of mu2 =', np.exp(mu2 + (mu2 - x_intersect))
    print 'percent of users on other side = ', len(feature_sum[feature_sum > \
                                        np.exp(mu2 + (mu2 - x_intersect))]) / \
                                        float(len(feature_sum)) * 100.0
    return [fig, x_intersect]



def plot_pca_components(X1, group, pca_params):
    x_min = pca_params['x_min']
    x_max = pca_params['x_max']
    x_delta = pca_params['x_delta']
    y_min = pca_params['y_min']
    y_max = pca_params['y_max']
    y_delta = pca_params['y_delta']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))
    ax1.plot(X1[:,0], X1[:,1], 'bo', alpha=0.6)
    ax2.plot(X1[:,0], X1[:,1], 'bo', alpha=0.6)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_xticks(np.arange(x_min, x_max + x_delta, x_delta))
    ax2.set_yticks(np.arange(y_min, y_max + y_delta, y_delta))

    title = 'Subgroup PCA components (group = %s)' % group
    ax1.set_title(title, fontsize=16)
    ax2.set_title('Zoomed In', fontsize=16)
    ax1.set_xlabel('X component', fontsize=14)
    ax2.set_xlabel('X component', fontsize=14)
    ax1.set_ylabel('Y component', fontsize=14)
    ax2.set_ylabel('Y component', fontsize=14)
    ax1.grid(True)
    ax2.grid(True)
    return fig


def remove_group_outliers(df_group, X1, group, outlier_params):
    x_min = outlier_params['x_min']
    x_max = outlier_params['x_max']
    y_min = outlier_params['y_min']
    y_max = outlier_params['y_max']

    df_use = df_group[(X1[:,0] >= x_min) & \
                    (X1[:,0] <= x_max) & \
                    (X1[:,1] >= y_min) & \
                    (X1[:,1] <= y_max)]
    df_use_rm = df_group[(X1[:,0] < x_min) | \
                        (X1[:,0] > x_max) | \
                        (X1[:,1] < y_min) | \
                        (X1[:,1] > y_max)]
    X2 = X1[(X1[:,0] >= x_min) & \
                    (X1[:,0] <= x_max) & \
                    (X1[:,1] >= y_min) & \
                    (X1[:,1] <= y_max)]
    X2_rm = X1[(X1[:,0] < x_min) | \
                        (X1[:,0] > x_max) | \
                        (X1[:,1] < y_min) | \
                        (X1[:,1] > y_max)]

    print 'original df_group =', df_group.shape
    print 'new df_use =', df_use.shape
    print 'new df_use_rm =', df_use_rm.shape
    print 'X2 =', X2.shape
    print 'X2_rm =', X2_rm.shape

    assert X2.shape[0] == df_use.shape[0], 'ERROR: Some dimension is off'
    return [X2, X2_rm, df_use, df_use_rm]



# ============================= EXAMINE SUBGROUPS ============================ #
'''
This function finds where the two subgroups overlap.
INPUTS:
x_vals          = The np.linspace() used for the plot
a1              = Altitude of first subgroup
mu1             = The mean of first subgroup
sigma1          = The standard deviation of the first subgroup
a2              = Altitude of second subgroup
mu2             = The mean of second subgroup
sigma2          = The standard deviation of the second subgroup
OUTPUT:
x_intersect     = The x-coordinate of the intersection point
'''
def get_x_intersect_t(mu1, sigma1, mu2, sigma2, deg_free):
    x_vals = np.linspace(mu1, mu2, 1000)
    f1 = stats.t.pdf(x_vals, loc=mu1, scale=sigma1, df=deg_free)
    f2 = stats.t.pdf(x_vals, loc=mu2, scale=sigma2, df=deg_free)
    delta = np.abs(f1-f2)
    idx = np.where(delta==np.min(delta))[0][0]
    x_intersect = x_vals[idx]
    return x_intersect


'''
This function is used to evaluate a bimodal distrubition of user characteristics.
In this case it looks at two subgroups of users based on feature_sum.
INPUTS:
feature_sum     = The sum of numeric feature counts (no timestamps)
a1              = Altitude of first subgroup
mu1             = The mean of first subgroup
sigma1          = The standard deviation of the first subgroup
a2              = Altitude of second subgroup
mu2             = The mean of second subgroup
sigma2          = The standard deviation of the second subgroup
OUTPUT:
fig             = The resulting figure
'''
def show_sub_groupings_t(feature_sum, subgroups_params, deg_free):
    h1 = subgroups_params['h1']
    mu1 = subgroups_params['mu1']
    sigma1 = subgroups_params['sigma1']
    h2 = subgroups_params['h2']
    mu2 = subgroups_params['mu2']
    sigma2 = subgroups_params['sigma2']
    x_max = subgroups_params['x_max']
    bin_width = subgroups_params['bin_width']

    feature_sum[feature_sum<=0] = 0.001 #don't want NaNs from log
    fig, ax = plt.subplots(figsize=(15,8))
    x_vals = np.linspace(0,x_max,1000)
    f1 = h1*stats.t.pdf(x_vals, loc=mu1, scale=sigma1, df=deg_free)
    f2 = h2*stats.t.pdf(x_vals, loc=mu2, scale=sigma2, df=deg_free)
    ax.hist(feature_sum, bins=np.arange(0,x_max,bin_width), align='right', \
                                                        label='Raw Activity')
    ax.plot(x_vals, f1, 'r-', lw=2, label='Secondary Group')
    ax.plot(x_vals, f2, 'c-',lw=2, label='Main Group')
    plt.vlines(x=mu1, ymin=0, ymax=h1*stats.t.pdf(mu1, loc=mu1, scale=sigma1, \
                            df=deg_free), lw=2.0, color='red', linestyles='--')
    plt.vlines(x=mu2, ymin=0, ymax=h2*stats.t.pdf(mu2, loc=mu2, scale=sigma2, \
                            df=deg_free), lw=2.0, color='cyan', linestyles='--')
    #ax.plot(x_vals,h1*mlab.normpdf(x_vals, mu1, sigma1), 'r-', lw=2, \
                                                #label='Secondary Group')
    #ax.plot(x_vals,h2*mlab.normpdf(x_vals, mu2, sigma2), 'c-',lw=2, \
                                                #label='Main Group')
    #plt.vlines(x=mu1, ymin=0, ymax=h1*mlab.normpdf(mu1, mu1, sigma1), \
                                    #lw=2.0, color='red', linestyles='--')
    #plt.vlines(x=mu2, ymin=0, ymax=h2*mlab.normpdf(mu2, mu2, sigma2), \
                                    #lw=2.0, color='cyan', linestyles='--')

    ax.set_title('Raw Activity Counts Distribution', fontsize=16)
    ax.set_xlabel('Log of the sum of features', fontsize=14)
    ax.set_ylabel('User count', fontsize=14)

    x_intersect = get_x_intersect_t(mu1, sigma1, mu2, sigma2, deg_free)
    y1 = h1*stats.t.pdf(x_intersect, loc=mu1, scale=sigma1, df=deg_free)
    y2 = h2*stats.t.pdf(x_intersect, loc=mu2, scale=sigma2, df=deg_free)
    ax.plot(x_intersect, y1, color='gold', marker='o', ms=8)
    ax.plot(x_intersect, y2, color='gold', marker='o', ms=8)
    plt.vlines(x=x_intersect, ymin=y1, ymax=y2, lw=2.0, color='gold', \
                                linestyles='-', label='Equal Probability')
    ax.legend(loc=0, labelspacing=1.5, fontsize=12)
    return [fig, x_intersect]


def get_group_labels_t(feature_sum, users, norm_params, deg_free):
    mu1 = norm_params['mu1']
    sigma1 = norm_params['sigma1']
    mu2 = norm_params['mu2']
    sigma2 = norm_params['sigma2']

    feature_sum[feature_sum<=0] = 0.001 #don't want NaNs from log
    log_feature_sum = np.log(feature_sum)
    prob1 = stats.t.pdf(log_feature_sum, loc=mu1, scale=sigma1, df=deg_free)
    prob2 = stats.t.pdf(log_feature_sum, loc=mu2, scale=sigma2, df=deg_free)
    #prob1 = mlab.normpdf(log_feature_sum, mu1, sigma1)
    #prob2 = mlab.normpdf(log_feature_sum, mu2, sigma2)
    df_prob = pd.DataFrame({'user_id':users, \
                            'log_feature_sum':log_feature_sum, \
                            'prob1':prob1, \
                            'prob2':prob2})
    df_prob['group'] = np.where(df_prob['prob1'] > df_prob['prob2'], 11, 12)
    return df_prob


def plot_sub_groups_gamma(feature_sum, subgroups_params, deg_free):
    h1 = subgroups_params['h1']
    a1 = subgroups_params['a1']
    b1 = subgroups_params['b1']
    c1 = subgroups_params['c1']
    h2 = subgroups_params['h2']
    mu2 = subgroups_params['mu2']
    sigma2 = subgroups_params['sigma2']
    x_max = subgroups_params['x_max']
    bin_width1 = subgroups_params['bin_width1']
    bin_width2 = subgroups_params['bin_width2']
    feature_sum[feature_sum<=0] = 0.001 #don't want NaNs from log
    x_vals = np.linspace(0,x_max,1000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,12))

    #Try a gamma distribution
    f1 = h1*stats.gamma.pdf(x_vals, a1, loc=b1, scale=c1)
    ax1.hist(feature_sum, bins=np.arange(0,x_max,bin_width1), align='right', \
                                                        label='Raw Activity')
    ax1.plot(x_vals, f1, 'r-', lw=4, label='Gamma\nDistribution')
    ax1.vlines(x=a1*c1, ymin=0, ymax=h1*stats.gamma.pdf(a1*c1, a1, loc=b1, scale=c1), \
                            lw=3.0, color='red', linestyles='--', label='Mean')
    ax1.set_title('Raw Activity Counts Distribution', fontsize=16)
    ax1.set_xlabel('Log of the sum of features', fontsize=14)
    ax1.set_ylabel('User count', fontsize=14)
    ax1.legend(loc=0, labelspacing=1.5, fontsize=12)

    # Try a normal distribution
    f2 = h2*stats.t.pdf(x_vals, loc=mu2, scale=sigma2, df=deg_free)
    ax2.hist(feature_sum, bins=np.arange(0,x_max,bin_width2), align='right', \
                                                        label='Raw Activity')
    ax2.plot(x_vals, f2, 'c-',lw=4, label='Gaussian\nDistribution')
    ax2.vlines(x=mu2, ymin=0, ymax=h2*stats.t.pdf(mu2, loc=mu2, scale=sigma2, \
            df=deg_free), lw=3.0, color='cyan', linestyles='--', label='Mean')

    ax2.set_title('Raw Activity Counts Distribution', fontsize=16)
    ax2.set_xlabel('Log of the sum of features', fontsize=14)
    ax2.set_ylabel('User count', fontsize=14)
    ax2.legend(loc=0, labelspacing=1.5, fontsize=12)
    return fig


'''
This function finds where the two subgroups overlap.
INPUTS:
x_vals          = The np.linspace() used for the plot
a1              = Altitude of first subgroup
mu1             = The mean of first subgroup
sigma1          = The standard deviation of the first subgroup
a2              = Altitude of second subgroup
mu2             = The mean of second subgroup
sigma2          = The standard deviation of the second subgroup
OUTPUT:
x_intersect     = The x-coordinate of the intersection point
'''
def get_x_intersect_gamma(a1, b1, a2, b2):
    x_vals = np.linspace(a1/b1, a2/b2, 10000)
    f1 = stats.gamma.pdf(x_vals, a1, b1)
    f2 = stats.gamma.pdf(x_vals, a2, b2)
    delta = np.abs(f1-f2)
    idx = np.where(delta==np.min(delta))[0][0]
    x_intersect = x_vals[idx]
    return x_intersect



'''
This function is used to evaluate a bimodal distrubition of user characteristics.
In this case it looks at two subgroups of users based on feature_sum.
INPUTS:
feature_sum     = The sum of numeric feature counts (no timestamps)
a1              = Altitude of first subgroup
mu1             = The mean of first subgroup
sigma1          = The standard deviation of the first subgroup
a2              = Altitude of second subgroup
mu2             = The mean of second subgroup
sigma2          = The standard deviation of the second subgroup
OUTPUT:
fig             = The resulting figure
'''
def show_sub_groupings_gamma(feature_sum, gamma_params):
    h1 = gamma_params['h1']
    a1 = gamma_params['a1']
    b1 = gamma_params['b1']
    h2 = gamma_params['h2']
    a2 = gamma_params['a2']
    b2 = gamma_params['b2']
    x_max = gamma_params['x_max']
    bin_width = gamma_params['bin_width']

    feature_sum[feature_sum<=0] = 0.001 #don't want NaNs from log (shouldn't happen)
    fig, ax = plt.subplots(figsize=(15,8))
    x_vals = np.linspace(0,x_max,1000)
    #f1 = a1*stats.t.pdf(x_vals, loc=mu1, scale=sigma1, df=deg_free)
    f1 = h1*stats.gamma.pdf(x_vals, a1, b1)
    f2 = h2*stats.gamma.pdf(x_vals, a2, b2)
    ax.hist(np.log(feature_sum), bins=np.arange(0,x_max,bin_width), align='right', \
                                                        label='Raw Activity')
    ax.plot(x_vals, f1, 'r-', lw=2, label='Secondary Group')
    plt.vlines(x=a1/b1, ymin=0, ymax=h1*stats.gamma.pdf(a1/b1, a1, b1), \
                        lw=2.0, color='red', linestyles='--', label='Mean')
    ax.plot(x_vals, f2, 'c-',lw=2, label='Main Group')
    plt.vlines(x=a2/b2, ymin=0, ymax=h2*stats.gamma.pdf(a2/b2, a2, b2), \
                        lw=2.0, color='red', linestyles='--', label='Mean')

    ax.set_title('Raw Activity Counts Distribution', fontsize=16)
    ax.set_xlabel('Log of the sum of features', fontsize=14)
    ax.set_ylabel('User count', fontsize=14)

    x_intersect = get_x_intersect_gamma(a1, b1, a2, b2)
    y1 = h1*stats.gamma.pdf(x_intersect, a1, b1)
    y2 = h2*stats.gamma.pdf(x_intersect, a2, b2)
    ax.plot(x_intersect, y1, color='gold', marker='o', ms=8)
    ax.plot(x_intersect, y2, color='gold', marker='o', ms=8)
    plt.vlines(x=x_intersect, ymin=y1, ymax=y2, lw=2.0, color='gold', \
                                linestyles='-', label='Equal Probability')
    ax.legend(loc=0, labelspacing=1.5, fontsize=12)
    return [fig, x_intersect]


def get_group_labels_gamma(feature_sum, users, gamma_params):
    a1 = gamma_params['a1']
    b1 = gamma_params['b1']
    a2 = gamma_params['a2']
    b2 = gamma_params['b2']
    log_feature_sum = np.log(feature_sum)
    prob1 = stats.gamma.pdf(log_feature_sum, a1, b1)
    prob2 = stats.gamma.pdf(log_feature_sum, a2, b2)
    df_prob = pd.DataFrame({'user_id':users, \
                            'log_feature_sum':log_feature_sum, \
                            'prob1':prob1, \
                            'prob2':prob2})
    df_prob['group'] = np.where(df_prob['prob1'] > df_prob['prob2'], 11, 12)
    #ensure that the gamma shift doesn't mess up groupings
    df_prob.loc[:, 'group'] = df_prob['group'].where(df_prob['log_feature_sum'] >= a/b, 11)
    return df_prob



# =============================== EVALUATION ================================= #


def get_cluster_users(df_use, lbl_vec, K):
    cluster_users = []
    for k in xrange(K):
        tmp_cluster_df = df_use[lbl_vec==k]
        cluster_users.append(tmp_cluster_df)
    return cluster_users



'''
This function gives the significance for each feature column between all
clusters.
INPUTS:
df_clusters     = A list of data frames containing the columns of interest for
                  the resulting clusters.
                  df_clusters = [pd.DataFrame(X[lbl_vec==0]), \
                                 pd.DataFrame(X[lbl_vec==1]), \
                                 pd.DataFrame(X[lbl_vec==2])]
plot_it         = If true, plot a histogram of the significane of each column
verbose         = If true, print the significance for each column
OUTPUT:
This function prints as it goes and shows the plot if plot_it==True
'''
def get_cluster_signif(df_clusters, plot_it=False, verbose=False):
    idx = range(len(df_clusters))
    combos = itertools.combinations(idx, 2)
    for combo in combos:
        a = combo[0]
        b = combo[1]
        df_a = df_clusters[a]
        df_b = df_clusters[b]
        print 'Cluster %d with cluster %d\n' % (a, b)
        print 'Feature | *p*-value'
        print '------- | -------'
        signif = []
        for c in df_a.columns:
            arr_a = np.array(df_a[c])
            arr_b = np.array(df_b[c])
            arr_a = arr_a[~np.isnan(arr_a)]
            arr_b = arr_b[~np.isnan(arr_b)]
            p = stats.ttest_ind(arr_a, arr_b, equal_var=False)[1]
            if np.isnan(p):
                p = 0.0 #e.g. all day_deltas == 0
            signif.append(p)
            if verbose:
                print '%s | %.4f' % (c, p)

        mean_signif = np.nanmean(signif)
        med_signif = np.percentile(signif, [50])[0]

        print '\nmean *p*-value | median *p*-value'
        print '------------ | --------------'
        print '%15.4f | %15.4f\n' % (mean_signif, med_signif)
        print '-' * 60
        print ''

        if plot_it:
            fig, ax = plt.subplots()
            ax.hist(signif, bins=np.arange(0, 1.05, 0.05))
            t1 = 'Significance of Clusters %d and %d' % (a, b)
            t2 = '\nMean=%.4f, Median=%.4f' % (mean_signif, med_signif)
            ax.set_title(t1+t2)
            ax.set_xlabel('p-value')
            ax.set_ylabel('Count of columns')
            plt.show()




