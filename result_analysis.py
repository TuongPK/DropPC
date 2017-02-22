import glob 
import cPickle
import gzip
import numpy
import matplotlib.pyplot as plt
import numpy as np


path = './Result/Sparsity/*.pkl'   

data = []
files=sorted(glob.glob(path))

for result in files:
    with open(result, 'rb') as f:
        output = cPickle.load(f)
        f.close()
    
    print result
    temp = [min(output[i][1]) for i in xrange(10)]
    print numpy.min(temp)
    print numpy.average(temp)
    data.append(temp)
    

plt.boxplot(data)   
plt.xticks([1, 2, 3, 4, 5, 6], ['AL_NFD', 'AL_nDO', 'NL_NFD', 'NL_nDO', 'PL_NFD', 'PL_nDO'])
plt.show()
