'''
This module works in conjunction with webcrawler.py and uses the results
created by webcrawler.py as input, viz. a directory called tickets/ with
text files corresponding to Fusion tickets. Therefore you must run 
webcrawler.py first. Edit the infile name in the main function at the bottom
of this file to correspond with the input file containing the ticket numbers.
INPUT
No direct input - edit the Control Panel at the bottom of this file to
give the input file with the ticket numbers and tags, the output file
desired (will be tab delimited) and specifiy whether the input file has
a header or not.

OUTPUT
A tab delimited file with the summary information from the fusion tickets.
Each line contains:
ticket              = The ticket number
company             = The Salesforce company name 
overview            = An brief overview of the problem (if given) 
date                = The date of the issue (if given) 
domains             = A list of domains impacted (if given) 
ip_affected         = A list of IPs impacted (if given)
resolution          = A brief summary of the ticket resolution 
tag                 = Fusion tag applied 
other_ip_addresses  = Any other IP addresses found in the ticket. These could
                      be relevant or not (may be RP addresses or other)

NOTE: Verify that line 36 has the right suffix, i.e. .txt or nothing.
'''


import os
import sys
import re

def extract_ticket_info(ticket, tag):
    inlog = open('tickets/' + ticket + '.txt', 'rt')
    lines = inlog.readlines()
    inlog.close()

    for idx,line in enumerate(lines):
        if 'Action not Permitted' in line:
            return None
        if 'Quick Filter' in line:
            idx1 = idx + 2
        if 'Resolution Summary' in line:
            idx2 = idx
        if 'Issue Start Date' in line:
            idx3 = idx
        if 'Sending Domain(s)' in line:
            idx4 = idx
        if 'IP Address(es) Impacted' in line:
            idx5 = idx

    #idx1 = [i for i, s in enumerate(lines) if 'Quick Filter' in s][0] + 2
    try:
        data1 = lines[idx1]
        company = re.search('Salesforce Account(.*?)function', data1).group(1)
        company = company.strip()[:-2] #remove whitespace and $( from $(function
    except:
        company = 'Unavailable'

    try:
        overview = re.search('Client Success(.*?)Salesforce', data1).group(1)
    except:
        pass
    try:
        overview = re.search('Client Support(.*?)Salesforce', data1).group(1)
    except:
        pass
    try:
        overview = re.search('Channel Support(.*?)Salesforce', data1).group(1)
    except:
        pass
    try:
        overview
    except NameError:
        overview = 'Unavailable'

    #idx2 = [i for i, s in enumerate(lines) if 'Resolution Summary' in s][0]
    try:
        data2 = lines[idx2]
        resolution = re.search('(.*?)Resolution Summary', data2).group(1)
    except:
        resolution = 'Unavailable'

    #idx3 = [i for i,s in enumerate(lines) if 'Issue Start Date' in s][0]
    try:
        data3 = lines[idx3]
        date = re.search('(.*)Issue Start Date(.*)Issue', data3).group(2)
    except:
        date = 'Unavailable'

    #idx4 = [i for i,s in enumerate(lines) if 'Sending Domain(s)' in s][0]
    try:
        data4 = lines[idx4]
        domains = re.search('(.*)Sending Domain\(s\) Impacted(.*)Mail', \
                                                            data4).group(2)
    except:
        domains = 'Unavailable'

    #idx5 = [i for i,s in enumerate(lines) if 'IP Address(es) Impacted' in s][0]
    try:
        data5 = lines[idx5:(idx4+1)]
        data5b = [i.strip() for i in data5]
        data5 = ', '.join(data5b)
        ip_affected = re.search('(.*)IP Address\(es\) Impacted(.*)Sending', \
                                                            data5).group(2)
    except:
        ip_affected = 'Unavailable'
    
    other_ip_addresses = []
    for line in lines:
        #other_ip_addresses.extend(re.findall( r'[0-9]+(?:\.[0-9]+){3}', line ))
        s = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        other_ip_addresses.extend(re.findall(s, line ))
    other_ip_addresses = list(set(other_ip_addresses))

    #The first is internal, the second looks like an ip, but is a version number
    #the next two are Return Path, the next shows up in every ticket and
    #is related to the customer service representative.
    #the third row are IPs belonging to mxlogic.net, a McAfee company
    #that has something to do with security
    extraneous_ips = ['255.255.255.255', '4.57.1.36', \
                    '10.0.1.142', '50.201.69.7', '50.201.69.34',
                    '208.65.144.247','208.65.144.245','208.65.145.245']
    for e in extraneous_ips:
        try:
            other_ip_addresses.remove(e)
        except:
            pass
    other_ip_addresses = ', '.join(other_ip_addresses)

    return [ticket, company, overview, date, domains, ip_affected, \
            resolution, tag, other_ip_addresses]
   
   
def print_results(info):
    print '\n-------------------------------------------------------------'
    print 'Ticket Number =', info[0]
    print 'Salesforce Account = ', info[1]
    print 'Overview = ', info[2]
    print 'Date = ', info[3]
    print 'Domains Impacted = ', info[4]
    print 'IPs Impacted = ', info[5]
    print 'Resolution = ', info[6]
    print 'Tag = ', info[7]
    print 'IP addresses = ', info[8]
    print '-------------------------------------------------------------'


def write_results(info, outlog):
    outlog.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % \
                (info[0], info[1], info[2], info[3], info[4], \
                info[5], info[6], info[7], info[8]))


if __name__ == '__main__':
    # ======================================================================== #
    #                           Control Panel                                  #
    # ======================================================================== #
    infile = 'tickets_tst.csv'
    outfile = 'output.tsv'
    header = True
    # ======================================================================== #
    
    inlog = open(infile, 'rt')
    if header:
        lines = inlog.readlines()[1:] #leave off heading
    else:
        lines = inlog.readlines()
    inlog.close()

    outlog = open(outfile, 'wt')
    outlog.write('Ticket\tCompany\tOverview\tDate\tDomains Impacted\t' + \
                'IPs Impacted\tResolution\tTag\tOther IPs Found\n')
    not_permitted = []
    for line in lines:
        [ticket, tag] = line.split(',')[:2]
        info = extract_ticket_info(ticket, tag)
        if info:
            print_results(info)
            write_results(info, outlog)
        else:
            not_permitted.append(ticket)

    outlog.close()

    meta = open('meta.txt', 'wt')
    for t in not_permitted:
        meta.write('%s\n' % t)
    meta.close()


