'''
This module looks for matches between two lists of strings. It uses fuzzy
matching to find similar, if not identical, matches. The functionality
can be controlled via the "Control Panel" at the bottom of the file.
INPUTS:
infile1     = The first .csv file with the column of strings to test. The
              only requirement is that the column has a heading.
infile2     = The second .csv file with the column of strings to test. The
              only requirement is that the column has a heading.
s1_idx      = The column heading of the first column to use (infile1).
s2_idx      = The column heading of the second column to use (infile2).
threshold   = The percentage to use for the similarity matching. It is an 
              integer between 0 and 100. This can also be passed as a 
              command line argument.

OUTPUTS:
meta.txt    = A text file with the matches

USAGE:
Use either with or without the command line arguement
$ python fuzzy_str.py
$ python fuzzy_str.py <percentage>, e.g. $ python fuzzy_str.py 80

DEPENDENCIES:
pandas:
On Mac you need to get Xcode first and double check you have gcc 
(type $ gcc and it should say no input files) Then you can do 
$ sudo easy_install pandas
If you get an error like clang: error: unknown argument:, let me know

fuzzywuzzy:
$ sudo easy_install fuzzywuzzy (should work, but didn't try it)
Or
$ sudo easy_install pip
$ sudo pip install fuzzywuzzy
'''

import sys
import time
import pandas as pd
from fuzzywuzzy import fuzz


'''
This is the main function that finds and prints the matching strings.
INPUTS:
df1          = pandas dataframe for infile1
df2          = pandas dataframe for infile2
s1_idx      = the heading of the first column of strings (infile1)
s2_idx      = the heading of the second column of strings (infile2)
threshold   = the similarity threshold to use (int between 0 and 100). 
              Defaults to 85.
'''
def find_matches(df1, df2, s1_idx, s2_idx, threshold):
    l1 = df1[s1_idx]
    l1.dropna(axis=0, inplace=True)
    other_meta('list 1 has %d elements' % len(l1))
    print 'list 1 has %d elements' % len(l1)
    
    l2 = df2[s2_idx]
    l2.dropna(axis=0, inplace=True)
    other_meta('list 2 has %d elements' % len(l2))
    print 'list 2 has %d elements' % len(l2)
    
    other_meta('Using similarity threshold of %d%%' % threshold)
    print 'Using similarity threshold of %d%%' % threshold
    count = 0
    print '%-40s\t\t%-40s' % ('List 1', 'List 2')
    other_meta('%-40s\t%-40s' % ('List 1', 'List 2'))

    matches = []
    for s1 in l1:
        for s2 in l2:
            if fuzz.partial_ratio(s1.lower(), s2.lower()) >= threshold:
                matches.append((s1.strip(), s2.strip()))
                print '%-40s\t\t%-40s' % (s1.strip(), s2.strip())
                other_meta('%-40s\t%-40s' % (s1.strip(), s2.strip()))
                count += 1
    other_meta('Total Matches = %d' % count)
    print 'Total Matches = %d' % count
    return matches


'''
A meta.txt file is produced that contains the matching items. The next three
functions are responsible for printing the matches and the run times.
'''
def start_meta(start_str):
    outlog = open('meta.txt', 'wt')
    outlog.write('Started: %s\n\n' % time.strftime('%I:%M:%S %p', start_str))
    outlog.close()
    

def other_meta(content):
    outlog = open('meta.txt', 'at')
    outlog.write('%s\n' % content)
    outlog.close()


def end_meta(start, end):
    outlog = open('meta.txt', 'at')
    outlog.write('\nCompleted: %s\n' % time.strftime('%I:%M:%S %p', time.localtime()))
    outlog.write('Elapsed time = %.4f seconds\n' % (end-start))
    outlog.close()


if __name__ == '__main__':
    # ======================================================================== #
    #                           Control Panel                                  #
    # ======================================================================== #
    infile1 = 'peytons_company_list.csv'
    infile2 = 'full_training_jan1_jun30.csv'
    s1_idx = 'My180 List' #this is the column heading of company name in infile1
    s2_idx = 'realm_name' #this is the column heading of company name in infile2
    if len(sys.argv) == 2: #checks is command line args used
        threshold = int(sys.argv[1]) #if similarity threshold passed in
    else:
        threshold = 85 #default similarity threshold
    # ======================================================================== #
    
    
    # ------------------------------ Initialize ------------------------------ #
    start = time.time() #used for elapsed time
    start_str = time.localtime() #used for printing time
    print "Started:", time.strftime("%I:%M:%S %p", start_str)
    start_meta(start_str)
    
    # ------------------------------ Get Data -------------------------------- #
    data1 = pd.read_csv(infile1)
    data2 = pd.read_csv(infile2)

    # ----------------------------- Get Matches ------------------------------ #
    find_matches(data1, data2, s1_idx, s2_idx, threshold)

    # ------------------------------- Wrap Up -------------------------------- #
    end = time.time()
    print '\nElapsed time = %.4f seconds' % (end-start)
    end_meta(start, end)

