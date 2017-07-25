"""
This function examines the correlation between all columns of a dataset or
the columns of the dataset in relation to a target column. 
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def get_correlations(df, cols=None, method='pearson'):
    """
    This function computes a correlation matrix for the features given in cols
    Args:
        df          = The dataframe object with the data        
        cols        = A list of columns to include or all columns if None
        method      = One of {'pearson', 'kendall', 'spearman'}
    Returns:
        corr        = The resulting correlation dataframe for the columns given
    """
    # Compute the correlation matrix
    if cols is None:
        corr = df.corr(method=method)
    else:
        corr = df[cols].corr(method=method)
    return corr


def get_target_correlations(df, target_col, cols=None):
    """
    This function computes the correlation and p-value between each column
    in the DataFrame and the target column. 
    Args:
        df          = A Pandas DataFrame with the data
        target_col  = The target column to compare against
        cols        = A list of columns to include or all columns if None
    Returns:
        results     = A list of tuples, where each tuple contains
                      (column_name, correlation, p-value) when compared to the
                      target column. A two-tailed t-test is used.
    """
    results = [] # [(column, correlation, p-value), ...]
    for c in list(df.columns):
        if c == target_col:
            continue
        try:
            [corr, p] = pearsonr(df[c], df[target_col])
            results.append((c, corr, p))
        except TypeError:
            pass
    return results


def plot_correlations(df, cols=None):
    """
    This function plots a correlation matrix for the features identified in cols
    Args:
        df          = The dataframe object with the data        
        cols        = A list of columns to include or all columns if None
    Returns:
        fig         = The resulting figure with the correlation matrix for the
                      columns given
    """
    # Compute the correlation matrix (corr is type DataFrame)
    corr = get_correlations(df, cols)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up plot
    sns.set(style="darkgrid")
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title('Correlation between features', fontsize=18)
    #fig.tight_layout()

    # Plot it
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, square=True, ax=ax, 
                linewidths=.5, vmin=-1.0, vmax=1.0, cbar_kws={"shrink": .5})
    return fig
    
