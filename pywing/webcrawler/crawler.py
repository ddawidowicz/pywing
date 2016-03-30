'''
This program implements a basic web crawler that can deal with a login
page. The program requires the package mechanize, which can be installed
with pip, $sudo pip install mechanize.

Also in order to get the details for the page you need, you can visit
the page and right click on it and choose inspect element to see the html
code for that page. Specifically you want

From the login page:
1)  What are the fieldnames for the username and password. In this example,
    they are 'username' and 'password' although different pages may have
    different naming conventions.
2)  Is there a single form or multiple forms on the page? In this example,
    there is only one form and can be accessed with browser.select_form(nr=0),
    but if there are multiple forms you need to use the name of the form for
    the login, which you can get from the inspect element command and load like,
    browser.select_form(name="some_name")

On the page itself:
1)  You need to look for how to get to the logout page, e.g. is it a single
    click or multiple clicks? Do you have to answer any questions when logging
    out? This example shows two ways to logout.

5/25/14
'''

import mechanize

def login(LOGIN_URL, USERNAME, PASSWORD):
    #open a browser object or emulator or whatever it is
    browser = mechanize.Browser()
    
    #ignore the robots.txt
    browser.set_handle_robots(False)

    #open the login page
    browser.open(LOGIN_URL)

    #select the first form on the page (in my case there was only one)
    #You can also look at the html with right-click "inspect element"
    #and select the form by name, 
    #browser.select_form(name="some_name")
    browser.select_form(nr = 0)

    #enter the user's credentials
    browser.form['username'] = USERNAME
    browser.form['password'] = PASSWORD

    #submit as user
    browser.submit()
    
    #return the browser for use in reading the page you want
    return browser

def logout(browser, LOGOUT_URL):
    #open the logout url directly
    browser.open(LOGOUT_URL)

    #or follow the link to logout
    #browser.follow_link(text=logout_txt)

    #To test the logout you could try again to read the page
    #and make sure you are reading the login page. Uncomment to check,
    #page = browser.open(PAGE_URL)
    #html = page.read()
    #print html


def read_page(browser, PAGE_URL):
    #you are now logged in and can access the page you actually want
    page = browser.open(PAGE_URL)
    return page


def print_page(page, EXTRA):

    #Get the full html from the page you accessed 
    #html = page.read().decode("UTF-8")
    html = page.read()
    print html

    #The page object contains information about the header, the server
    #response code, the length of the page, etc. If the extra variable is
    #set to True the extra information will print also
    if EXTRA: 
        #The shows the page object
        print "Response:", page

        # Get the URL. This gets the real URL. 
        print "The URL is: ", page.geturl()

        # Getting the code returned by the remote server
        print "This gets the code: ", page.code

        # Get the Headers. 
        # This returns a dictionary-like object that describes the page fetched, 
        # particularly the headers sent by the server
        print "The Headers are: ", page.info()

        # Get the date part of the header
        print "The Date is: ", page.info()['date']

        # Get the server part of the header
        print "The Server is: ", page.info()['server']

        # Get only the length
        print "Get the length :", len(html)

if __name__ == '__main__':
    # ========================= Control Panel ================================ #
    USERNAME = 'demo' #this is the username to supply the login window
    PASSWORD = 'demo' #this is the password for the login window
    LOGIN_URL = "<login url here>"
    PAGE_URL = "<main_page_url_here>"
    LOGOUT_URL = "<logout url here>"
    EXTRA = False #display extra header information?
    LOGOUT_TXT = 'Logout'
    # ======================================================================== #

    browser = login(LOGIN_URL, USERNAME, PASSWORD)
    page = read_page(browser, PAGE_URL)
    print_page(page, EXTRA)
    logout(browser, LOGOUT_TXT)
