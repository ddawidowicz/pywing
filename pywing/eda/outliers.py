import numpy as np

def get_mild_iqr_outlier_bounds(df):
    """
    This function finds the "mild" outliers using the IQR.
    Reference: http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
    Args:
        df      = Pandas dataframe with the data
    Returns:
        lower   = the lower bound for outliers
        upper   = the upper bound for outliers
    """
    [lq, uq] = np.nanpercentile(df, [25, 75], axis=0)
    iqr = uq - lq
    lower = lq - (1.5 * iqr)
    upper = uq + (1.5 * iqr)
    return [lower, upper]


def get_extreme_iqr_outlier_bounds(df):
    """
    This function finds the "extreme" outliers using the IQR.
    Reference: http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
    Args:
        df      = Pandas dataframe with the data
    Returns:
        lower   = the lower bound for outliers
        upper   = the upper bound for outliers
    """
    [lq, uq] = np.nanpercentile(df, [25, 75], axis=0)
    iqr = uq - lq
    lower = lq - (3.0 * iqr)
    upper = uq + (3.0 * iqr)
    return [lower, upper]


def modified_zscore_outlier_bounds(df):
    """
    This function finds the bounds for outliers using a modified z-score method.
    Reference: http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    Args:
        df      = Pandas dataframe with the data
    Returns:
        lower   = the lower bound for outliers
        upper   = the upper bound for outliers
    """
    threshold = 3.5
    mult_constant = 0.6745 # see link in comment above

    median = df.median(axis=0).astype(float)
    mad = np.median(np.abs(df.subtract(median, axis=1)))
    modified_z_scores = (mult_constant * df.subtract(median, axis=1)) / mad
    lower = -(3.5 * mad - 0.6745*median) / 0.6745
    upper = (3.5 * mad + 0.6745*median) / 0.6745
    return [lower, upper]


def outlier_stats(df):
    lower, upper = get_outlier_bounds(df)
    prct_outliers = df[df


def visualize_outliers(df):
    pass
