import os
import time
#import subprocess
import pexpect


'''
process = subprocess.Popen(['/bin/bash'], shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
process.stdin.write('echo it works!\n')
print process.stdout.readline()
process.stdin.write('date\n')
print process.stdout.readline()

start = time.time() #used for elapsed time
start_str = time.localtime() #used for printing time
print 'Started: %s\n' % time.strftime('%I:%M:%S %p', start_str)

end = time.time()
print 'Completed: %s\n' % time.strftime('%I:%M:%S %p', time.localtime())
print 'Elapsed time = %.4f seconds\n' % (end-start)

#==============================================================================#

process = subprocess.Popen(['/usr/bin/hive'], shell=False, stdin=subprocess.PIPE)
process.stdin.write('desc jkb_oib_gmail_yahoo_domain;')
process.stdin.write('\x1B13')
process.stdin.write('quit;')
print process.stdout.readline()
#==============================================================================#

proc = subprocess.Popen(['/usr/bin/hive','show tables;'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
out = proc.communicate('')
print out
#==============================================================================#
'''
child = pexpect.spawn('hive')
#child.expect('hive (default)> ')
child.sendline('show tables;')
o2 = child.after
print 'after'
print o2
'''
child = pexpect.spawn('python')
child.logfile = open('mylog.txt','wb')
child.expect('>>>')
print '%d' % (2+2)
print('And now for something completely different...')
print(''.join(reversed((child.before))))
print('Yes, it\'s python, but it\'s backwards.')
print()
print('Escape character is \'^]\'.')
#print(c.after, end=' ')
#c.interact()
child.kill(1)
print('is alive:', child.isalive())
#child.send('2+2')
#o2 = child.after
#print 'after'
#print o2
>>>
'''


