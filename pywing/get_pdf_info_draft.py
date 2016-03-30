import pudb
'''
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from cStringIO import StringIO


def convert_pdf(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    #process_pdf(rsrcmgr, device, fp)
    PDFPage.get_pages(rsrcmgr, device, fp)
    fp.close()
    device.close()

    txt_str = retstr.getvalue()
    retstr.close()
    return txt_str
'''

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

def convert_pdf(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = file(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, \
                                password=password,caching=caching, \
                                check_extractable=True):
        interpreter.process_page(page)
    fp.close()
    device.close()
    str = retstr.getvalue()
    retstr.close()
    return str

def main():
    infiles = ['/Users/../tst1.pdf', \
                '/Users/.../tst2.pdf']
    for infile in infiles:
        txt_str = convert_pdf(infile)
        txt_list = txt_str.split('\n')
        txt_list = filter(None, txt_list)
        #print txt_list
        ascii_list = remove_non_ascii(txt_list)
        #print ascii_list

        #idx1 = ascii_list.index('ClientSuccess')
        print '--------------------------------------'
        print 'File = ', infile[21:]
        print 'Ticket =', ascii_list[1]
        idx = ascii_list.index('Resolution Summary')
        print 'Conclusion = %s' % ascii_list[idx-1]

def remove_non_ascii(txt_list):
    fixed = []
    for s in txt_list:
        s = s.split('\xa0')
        s = ' '.join(s)
        new_s = "".join(i for i in s if ord(i)<126 and ord(i)>31)
        fixed.append(new_s)
    return fixed

main()
