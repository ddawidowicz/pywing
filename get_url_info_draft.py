import urllib, urllib2, base64
import cookielib

url = '<page_url>'
username = '<user_name>'
password = '<password>'

passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
# this creates a password manager
passman.add_password(None, url, username, password)
# because we have put None at the start it will always
# use this username/password combination for  urls
# for which `url` is a super-url

authhandler = urllib2.HTTPBasicAuthHandler(passman)
# create the AuthHandler
opener = urllib2.build_opener(authhandler)

urllib2.install_opener(opener)
# All calls to urllib2.urlopen will now use our handler
# Make sure not to include the protocol in with the URL, or
# HTTPPasswordMgrWithDefaultRealm will be very confused.
# You must (of course) use it when fetching the page though.


response = urllib2.urlopen(url)
# authentication is now handled automatically for us

'''
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor())
urllib2.install_opener(opener)

#Next use the opener to POST to the login form and the protected page. 
#The cookie returned by the server will be captured by the HTTPCookieProcessor:
params = urllib.urlencode(dict(username='<uname_here>', password='<pw_here'))
response = opener.open('<login_url_here>', params)
data = response.read()
print 'data1'
print data
print '\n\n'
response.close()

response = opener.open('<page_url_here>')
data = response.read()
response.close()
print 'data2'
print data
'''

#url = '<page_url>'
#username = '<user_name>'
#password = '<password>'
#
#request = urllib2.Request(url)
#base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
#request.add_header("Authorization", "Basic %s" % base64string)   
#request.add_header('Content_Type','application/x-www-form-urlencoded;charset=UTF-8')
#response = urllib2.urlopen(request)

'''
passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
passman.add_password(None, url, username, password)
urllib2.install_opener(urllib2.build_opener(urllib2.HTTPBasicAuthHandler(passman)))
'''
req = urllib2.Request('<url_here>')
response = urllib2.urlopen(req)


#get the response from the server
print "Response:", response

# Get the URL. This gets the real URL. 
print "The URL is: ", response.geturl()

# Getting the code
print "This gets the code: ", response.code

# Get the Headers. 
# This returns a dictionary-like object that describes the page fetched, 
# particularly the headers sent by the server
print "The Headers are: ", response.info()

#Get the raw data from the page
data = response.read()
print 'Here is the data:'
print(data)

'''
#url_str = 'http://scikit-learn.org/stable/auto_examples/' + \
#            'document_clustering.html#example-document-clustering-py'

url_str = '<url_here>'
response = urllib2.urlopen(url_str)
print "Response:", response

# Get the URL. This gets the real URL. 
print "The URL is: ", response.geturl()

# Getting the code
print "This gets the code: ", response.code

# Get the Headers. 
# This returns a dictionary-like object that describes the page fetched, 
# particularly the headers sent by the server
print "The Headers are: ", response.info()

# Get the date part of the header
print "The Date is: ", response.info()['date']

# Get the server part of the header
print "The Server is: ", response.info()['server']

# Get all data
html = response.read()
print "Get all data: ", html

# Get only the length
print "Get the length :", len(html)

# Showing that the file object is iterable
#for line in response:
#    print line.rstrip()

# Note that the rstrip strips the trailing newlines and carriage returns before
# printing the output.

'''

