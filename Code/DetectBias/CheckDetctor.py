
import numpy as np
from Detector_WeightedAddition import Detect
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

data = np.genfromtxt('data\student-por_Cleaned.csv',delimiter=',',usecols=np.arange(0,46))
#data = np.genfromtxt('data\crimecommunity.csv',delimiter=',',usecols=np.arange(0,102))
data = np.nan_to_num(data)
sample = data[1:,:]
[n,p] = sample.shape
label = sample[:,0]
feature_start = 17
feature_end = 46
#max_depth = feature_end - feature_start
max_depth = 31
threshold = 0.3

print('detecting...')
Feature = Detect(sample,max_depth,threshold,feature_start,feature_end)
print('done detecting')
num_feature = len(Feature)

totalDP = np.zeros([num_feature,num_feature])

for i in range(num_feature):
    for j in range(num_feature):
        print('processing: {} and {}'.format(i,j))
        num_11 = len([e for e in range(n) if sample[e,Feature[i]] == 1 and sample[e,Feature[j]] == 1])
        num_00 = len([e for e in range(n) if sample[e,Feature[i]] == 0 and sample[e,Feature[j]] == 0])
        num_01 = len([e for e in range(n) if sample[e,Feature[i]] == 0 and sample[e,Feature[j]] == 1])
        num_10 = len([e for e in range(n) if sample[e,Feature[i]] == 1 and sample[e,Feature[j]] == 0])
        
        num_11_dropout = len([e for e in range(n) if label[e] == 0 and sample[e,Feature[i]] == 1 and sample[e,Feature[j]] == 1])
        num_00_dropout = len([e for e in range(n) if label[e] == 0 and sample[e,Feature[i]] == 0 and sample[e,Feature[j]] == 0])
        num_01_dropout = len([e for e in range(n) if label[e] == 0 and sample[e,Feature[i]] == 0 and sample[e,Feature[j]] == 1])
        num_10_dropout = len([e for e in range(n) if label[e] == 0 and sample[e,Feature[i]] == 1 and sample[e,Feature[j]] == 0])
        
#        if num_11 != 0 and num_00 != 0 and num_01 != 0 and num_10 != 0:
#            demographic_parity_11 = (abs(num_11_dropout/num_11 - (num_00_dropout+num_01_dropout+num_10_dropout)/(num_00+num_01+num_10)))
#            demographic_parity_00 = (abs(num_00_dropout/num_00 - (num_11_dropout+num_01_dropout+num_10_dropout)/(num_11+num_01+num_10)))
#            demographic_parity_01 = (abs(num_01_dropout/num_01 - (num_00_dropout+num_11_dropout+num_10_dropout)/(num_00+num_11+num_10)))
#            demographic_parity_10 = (abs(num_10_dropout/num_10 - (num_00_dropout+num_01_dropout+num_11_dropout)/(num_00+num_01+num_11)))
#            
        if (num_11 == 0):
            demographic_parity_11 = 0
        else:
            demographic_parity_11 = (abs(num_11_dropout/num_11 - (num_00_dropout+num_01_dropout+num_10_dropout)/(num_00+num_01+num_10)))
        
        if (num_00 == 0):
            demographic_parity_00 = 0
        else:
            demographic_parity_00 = (abs(num_00_dropout/num_00 - (num_11_dropout+num_01_dropout+num_10_dropout)/(num_11+num_01+num_10)))
#
        if (num_01 == 0):
            demographic_parity_01 = 0
        else:
            demographic_parity_01 = (abs(num_01_dropout/num_01 - (num_00_dropout+num_11_dropout+num_10_dropout)/(num_00+num_11+num_10)))
            
        if (num_10 == 0):
            demographic_parity_10 = 0
        else: 
            demographic_parity_10 = (abs(num_10_dropout/num_10 - (num_00_dropout+num_01_dropout+num_11_dropout)/(num_00+num_01+num_11)))
            
        
        maxDP = max(demographic_parity_11,demographic_parity_00,demographic_parity_01,demographic_parity_10)
        
        totalDP[i,j] = maxDP
        
viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)

plt.pcolormesh(totalDP,cmap=newcmp,rasterized=True,vmin=0,vmax=1)
plt.colorbar()
plt.show()