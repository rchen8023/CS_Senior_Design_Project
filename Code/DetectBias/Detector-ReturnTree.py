
import numpy as np
import math


data = np.genfromtxt('student-por_Cleaned.csv',delimiter=',',usecols=np.arange(0,46))
data = np.nan_to_num(data)

sample = data[1:,:]
[n,p] = sample.shape

feature_start = 17
feature_end = 46
#max_depth = feature_end - feature_start
max_depth = 7
threshold = 0.5


def node_feature(nodes,existFeatures,dn):
    entropy = float("inf")
    feature_index = -1

    for index in range(feature_start,feature_end):
        
        entropy_total = 0
        for i in range(dn):
            if getEntropy(nodes[i]) > threshold:
                
                if index not in existFeatures:
                    
                    left_temp,right_temp = split(index,nodes[i])
                    
                    num_left = len(left_temp)
                    num_right = len(right_temp)
                        
                    prob_feature_right = num_right/len(nodes[i])
                    prob_feature_left = num_left/len(nodes[i])
                    
                    right_entropy = getEntropy(right_temp)
                    left_entropy = getEntropy(left_temp)
                    
                    weighted_entropy = prob_feature_right*right_entropy + prob_feature_left*left_entropy
                    
                    entropy_total = entropy_total + weighted_entropy
                    
                    
                else:
                    entropy_total = entropy
                    break
        if entropy_total < entropy:
            entropy = entropy_total
            feature_index = index

        
    existFeatures.append(feature_index)
            
    return feature_index

def getEntropy(data):
    if len(data) == 0:
        return 0
    prob_0 = sum(i==0 for i in data[:,0])/len(data)
    prob_1 = sum(i==1 for i in data[:,0])/len(data)
    
    if prob_0 == 0 or prob_1 == 0:
        entropy = 0
    else:
        entropy = -prob_0*np.log2(prob_0) - prob_1*np.log2(prob_1)
    
    return entropy

def split(feature_index,dataset):
    left,right = list(),list()
    
    left_index = dataset[:,feature_index] == 1
        
    left = dataset[left_index]
    right = dataset[~left_index]
    
    return left,right

def BuildTree(node,existFeatures):
    Tree = [node]

    
    for i in range(max_depth):
        dn = 2**i 
        if i == max_depth-1:
            break
        
        current_leaves = Tree[dn-1:]
        index = node_feature(current_leaves,existFeatures,dn)
        
        if index == -1:
            break
        
        for j in range(dn):
            
            if getEntropy(current_leaves[j]) < 0.5:
                left = current_leaves[j]
                right = []
            else:
                left,right = split(index,current_leaves[j])

            
            Tree.append(left)
            Tree.append(right)
        
    return Tree,dn


Features = list()
sample_rate = 1
trails = 1
num_samples = math.floor(sample_rate * len(sample))
for i in range(trails):
    existFeatures = list()
    np.random.shuffle(sample)
    
    some_samples = sample[:num_samples,:]
    
    Tree,dn = BuildTree(some_samples,existFeatures)
    
    leaves = Tree[dn-1:]
    leaves_clean = [e for e in leaves if len(e) != 0]
    Features.append(existFeatures)
    

dropout_rate = []
entropy = []
for i in range(len(leaves)):
    leaf = leaves[i]
    if len(leaf)==0:
        dropout_rate.append(-1)
        entropy.append(-1)
    else:
        dropout_rate.append((len(leaf)-sum(leaf[:,0])) / len(leaf))
        entropy.append(getEntropy(leaf))
    

num_pure = len([e for e in range(len(entropy)) if entropy[e] <= 0.5 and entropy[e] != -1]) 
num_high_dropout = len([e for e in range(len(dropout_rate)) if dropout_rate[e] > 0.5])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
