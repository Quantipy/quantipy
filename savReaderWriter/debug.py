import sys
path = '/home/albertjan/nfs/Public/savreaderwriter'
sys.path.insert(0, path)
from savReaderWriter import *

savFileName = '/home/albertjan/nfs/Public/savreaderwriter/savReaderWriter/test_data/Employee data.sav'

with SavReader(savFileName) as reader:
    for line in reader:
        print(line)
        #pass