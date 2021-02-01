
import numpy as np
import math

data = np.genfromtxt('student-por_Cleaned.csv', delimiter=',', usecols=np.arange(0,46))
data = np.nan_to_num(data)

sample = data[1:,:]
[n,p] = sample.shape

max_depth = 4
depth = 1
index = 17

def test_split(index, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] == 1:
            left.append(row)
        elif row[index] == 0:
            right.append(row)
    return left, right

x,y = test_split(17,sample)

def get_label(dataset):
    label = list()
    for i in range(len(dataset)):
        temp_d = dataset[i]
        label.append(temp_d[0])
        
    return label

leaves = list()
def split(node,max_depth,depth,leaves,index):
    left,right = test_split(index,node)

    if depth >= max_depth-1:
#        leaves.append(get_label(left))
#        leaves.append(get_label(right))
        leaves.append(left)
        leaves.append(right)
        return
    else:
        split(left,max_depth,depth+1,leaves,index+1)
        split(right,max_depth,depth+1,leaves,index+1)

def define_feature(clusters):
    S = list()
#    S_n = list()
    for i in range(index,len(clusters[0])):
        feature = []
        entropy = 0
#        entropy_n = 0
        for j in range(len(clusters)):
            temp_c = clusters[j]
            feature.append(temp_c[i])

        prob = sum(feature) / len(feature)
        if prob == 0:
            entropy = None
        else:
            entropy = -(prob * math.log(prob))

        S.append(entropy)

    return S

split(sample,max_depth,depth,leaves,index)
purities = []
S = []
for i in range(len(leaves)):
    temp_l = leaves[i]
    label = get_label(temp_l)
    purities.append(sum(label)/len(label))
    S.append(define_feature(temp_l))

X = define_feature(leaves[3])
print(purities)
    

    
    