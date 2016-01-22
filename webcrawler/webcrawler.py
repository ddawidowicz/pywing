'''
This file scrapes tickets from Fusion and writes content to a text file for
further processing outside of Fusion.
Inputs (command line args):
1) USERNAME     = This is the username used to login
2) PASSWORD     = This is the password accompanying the username
3) TICKET_FILE  = This is the .csv file holding the ticket numbers (one per line)
Output:
outputs a text file for each ticket number given in the TICKET_FILE and written
to the subdirectory tickets/
'''

import sys
from crawler import login, read_page, logout
from parser import get_text


def read_tickets_from_file():
    # ========================= Control Panel ================================ #
    USERNAME = sys.argv[1] #this is the username to supply the login window
    PASSWORD = sys.argv[2] #this is the password for the login window
    TICKET_FILE = sys.argv[3] #this is the file with the ticket numbers
    LOGIN_URL = "<login url here>"
    PAGE_URL = "<main_page_url_here>"
    LOGOUT_URL = "<logout url here>"
    HEADING = True
    # ======================================================================== #

    c_t = sum(1 for line in open(TICKET_FILE, 'rt')) #total number of tickets
    c_i = 1 #a count to keep track by when retrieving

    inlog = open(TICKET_FILE, 'rt')
    if HEADING:
        inlog.next() #ignore heading on ticket file
        c_t -= 1 #remove heading from count
    
    browser = login(LOGIN_URL, USERNAME, PASSWORD) #create logged in browser obj
    for t in inlog:
        try:
            t = t.split(',')[0]
            t = t.strip() #remove newline character
            print 'Retrieving ticket %s (%d of %d)' % (t, c_i, c_t)
            page = read_page(browser, PAGE_URL + t) #retrieve page
            html = page.read() #extract html
            [txt_list, txt_str] = get_text(html) #clean html and return content

            #write ticket content to file
            outlog = open('tickets/' + t + '.txt', 'wt') 
            outlog.write(txt_str) 
            outlog.close()
        except:
            pass
        c_i += 1 #increment count
        
    inlog.close()

    #logout and verify logout
    logout(browser, LOGOUT_URL)
    page = read_page(browser, PAGE_URL)
    html = page.read()
    [_, txt_str] = get_text(html)
    if "pagetype = 'login'" in txt_str:
        print 'Logoff successful!'
    else:
        print txt_str


if __name__ == '__main__':
    read_tickets_from_file()
