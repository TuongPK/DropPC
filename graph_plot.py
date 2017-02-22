
# coding: utf-8

# In[37]:

import glob 
import cPickle
import gzip
import numpy
import theano
import matplotlib.pyplot as plt

path = './Result/*.pkl'   

files=glob.glob(path)   

for result in files: 
    with open(result, 'rb') as f:
        data = cPickle.load(f)
        
        train_data = data[0][0]
        valid_data = data[0][1]
        test_data = data[0][2]
        duration = data[0][3]
        f.close()
    
    plt.plot(valid_data)

plt.show()


# In[ ]:



