import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
mpl.use('TkAgg')
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

#half map
def plot_correlation(df_x, annotate=False):
    sns.set(style="white")
    # Compute the correlation matrix
    corr = df_x.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('Correlation between features', fontsize=16)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, annot=annotate,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    return fig



'''
# Full heatmap
corrmat = df2.iloc[:,1:-11].corr()

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(15, 15))

# Draw the heatmap using seaborn
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corrmat, vmax=.8, square=True, cmap=cmap)

# Use matplotlib directly to emphasize known networks
cols = list(corrmat.columns)
for i, col in enumerate(cols):
    if i and col != cols[i - 1]:
        ax.axhline(len(cols) - i, c="w")
        ax.axvline(i, c="w")
fig.tight_layout()
return fig
'''


