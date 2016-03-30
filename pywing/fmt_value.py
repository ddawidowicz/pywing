import locale
locale.setlocale(locale.LC_ALL, 'en_US')

'''
This function formats the data to have a comma between the thousands or millions
and to have only two decimal places.
INPUT:
y   = The number to be formated
OUTPUT:
The formatted number
'''
def fmt_value(y): 
    if isinstance(y, int):
        return locale.format('%d', y, grouping=True)
    else:
        return locale.format('%.2f', y, grouping=True)

