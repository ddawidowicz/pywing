'''
This module takes in a string extracted from a webpage with its html formatting
and returns both a list of lines without formatting and a single string 
without formatting. The function to call is get_text(html)

The main function to call is get_text(html)
Input:
html =  a single string with html formatting

Outputs:
txt_list =  A list of text lines without the html formatting
txt_str =   A single string of text without the html formatting
'''

import re
from HTMLParser import HTMLParser

'''
This class and the strip_tags(html) function below were taken from this post:
http://stackoverflow.com/questions/753052/strip-html-from-strings-in-python

They provide the functionality to remove html formatting from a strin of text.
'''
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

'''
This is just a wrapper around the class and function above. See the top
comments for usage.
'''
def get_text(html):
    #This is the string stripped of html formatting
    #It still may be sort of ugly because of newline characters (possibly many)
    orig_str = strip_tags(html)

    #split the string into a list at each newline character
    txt_list = re.split(r'[\n\r]+', orig_str)

    #strip any extraneous spaces from the elements
    txt_list = map(lambda x: x.strip(), txt_list)

    #remove any empty strings
    txt_list = filter(None, txt_list)

    #rejoin into a single string with newline characters
    txt_str = '\n'.join(txt_list)

    #return both the cleaned up list and the single string
    return [txt_list, txt_str]

