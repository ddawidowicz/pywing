import numpy as np
from datetime import datetime

'''
This function takes in an integer like 20140901, or an array of such integers
and creates a Python datetime object.
INPUT:
x       = An integer, or array of integers, that represent a date in the 
          form 20140901
OUTPUT:
x_dt    = A Python datetime object that represents the integer(s) passed in
'''
def get_date_obj(x):
    try:
        #list of values
        x_str = map(str, x)
        x_dt = map(lambda s: datetime(year=int(s[0:4]), month=int(s[4:6]), \
                            day=int(s[6:8])).date(), x_str)
        return np.array(x_dt)
    except:
        #single value
        x_str = str(x)
        x_dt = datetime(year=int(x_str[0:4]), month=int(x_str[4:6]), \
                            day=int(x_str[6:8])).date()
    return x_dt

