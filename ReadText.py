import sys
import re

def ReadText(text):
    #Be carefull! Selected text file shouldn't include empty row!
    text = "Setting/"+text
    datalist = {}
    num_lines = sum(1 for line in open(text))

    with open(text,'r') as f:
        line = f.readlines()
    
    for i, data in enumerate(line):
        datalist[re.findall('(.*):.*',data)[0]]=re.findall('.*:(.*)',data)[0]

    return datalist

if __name__=='__main__' :
    d=ReadText(sys.argv[1])
    print(d)
    
