import pandas as pd
import matplotlib.pyplot as plt

from correlations import *
from percentiles import *
from load_data import *


def run_correlations(df):
    corr = get_correlations(df)
    print(corr)

    target_col = 'sepal_width'
    corr_list = get_target_correlations(df, target_col)
    for item in corr_list:
        print('{} vs. {}:\tr-value = {:.4f}\tp-value = {:.4f}'
            .format(item[0], target_col, item[1], item[2]))

    fig = plot_correlations(df)
    plt.show()


def run_percentiles(df):
    print_percentiles(df)


if __name__ == '__main__':
    infile = '../data/iris.csv'
    df = load_csv_to_df(infile)

    #run_correlations(df)
    #run_percentiles(df)

