import numpy as np

def get_percentiles(df, cols=None, prct=None):
    """
    This function computes the percentiles given in the percentiles list.
    Args:
        df      = The Pandas DataFrame with the data
        cols    = A list of columns to consider or all columns if None
        prct    = A list of percentiles to calculate - a default set is used
                  if None
    Returns:
        results = A list of the computed percentiles
        prct    = The percentiles calculated
    """
    if not prct:
        prct = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    if cols is None:
        results = np.nanpercentile(df, prct, axis=0)
    else:
        results = np.nanpercentile(df[cols], prct, axis=0)
    return [results, prct]


def print_percentiles(df, cols=None, prct=None):
    """
    This function prints the results of the get_percentiles() function.
    Args:
        df      = The Pandas DataFrame with the data
        cols    = A list of columns to consider or all columns if None
        prct    = A list of percentiles to calculate - a default set is used
                  if None
    Returns:
        results = A list of the computed percentiles
        prct    = The percentiles calculated
    """
    # Choose only the numeric columns
    if cols is None:
        df = df.select_dtypes(['number'])
    else:
        df = df[cols].select_dtypes(['number'])
    cols = df.columns
    
    results, prct = get_percentiles(df, cols=cols, prct=prct)
    for c_idx, c in enumerate(cols):
        print('-' * 80)
        print(c)
        print('-' * 80)
        for r_idx, p in enumerate(prct):
            print('{:.3f} percentile = {:.4f}'.format(p, results[r_idx, c_idx]))
