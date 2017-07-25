import pandas as pd

def load_csv_to_df(infile, header=True):
    """
    This function loads data from a .csv file. Data is returned as a Pandas
    DataFrame
    """
    df = pd.read_csv(infile)
    return df
