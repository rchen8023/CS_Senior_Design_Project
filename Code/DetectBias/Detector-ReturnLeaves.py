
import numpy as np
import math


data = np.genfromtxt('student-por_Cleaned.csv',delimiter=',',usecols=np.arange(0,46))
data = np.nan_to_num(data)

sample = data[1:,:]
[n,p] = sample.shape

feature_start = 17
feature_end = 46
#max_depth = feature_end - feature_start
max_depth = 6
threshold = 0.5
trails = 1


Entropy = list()

def node_feature(nodes,existFeatures,dn):
    entropy = float("inf")
    feature_index = -1
#    left,right = list(),list()
    
    for index in range(feature_start,feature_end):
        
        entropy_total = 0
#        pure = False
        for i in range(dn):
            
            if getEntropy(nodes[i]) > threshold:
                
                if index not in existFeatures:
    #                if getEntropy(nodes[dn+i-1]) == 0:
    #                    pure = True
    #                    print(["break from 1, index is ", index, " nodes = ", dn+i-1])
    #                    break
                    
                    left_temp,right_temp = split(index,nodes[i])
                    
                    num_left = len(left_temp)
                    num_right = len(right_temp)
                    
#                    if num_left != 0 and num_right != 0:
                        
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
#        if pure == True:
#            feature_index = -1
#            print(["break from 4, index is ", index])
#            break
        
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
    nodes = [node]
    leaves = list()
#    nodes.append(node)
    
    for i in range(max_depth):
        temp_nodes = list()
#        dn = 2**i 
        dn = len(nodes) # number of impure nodes in current depth
        index = node_feature(nodes,existFeatures,dn)
        
        if index == -1:
#            leaves = nodes[dn-1:]
            break
        
        for j in range(dn):
            
            left,right = split(index,nodes[j])
#            
#            Entropy.append(getEntropy(left))
#            Entropy.append(getEntropy(right))
#            nodes.append(left)
#            nodes.append(right)
            
            left_entropy = getEntropy(left)
            right_entropy = getEntropy(right)
            
            if left_entropy <= threshold:
                
                leaves.append(left)
            else:
                temp_nodes.append(left)
                
            if right_entropy <= threshold:
                leaves.append(right)
            else:
                temp_nodes.append(right)
            
            
            
        
        nodes = temp_nodes
        
    leaves.extend(temp_nodes)
    leaves_x = [e for e in leaves if len(e) != 0]
#    leaves_x = leaves
    return leaves_x, temp_nodes


Features = list()
sample_rate = 1
trails = 1
num_samples = math.floor(sample_rate * len(sample))
for i in range(trails):
    existFeatures = list()
#    np.random.shuffle(sample)
    
    some_samples = sample[:num_samples,:]
    
    leaves,temp_nodes = BuildTree(some_samples,existFeatures)
    
    Features.append(existFeatures)
    
#leaves,temp_nodes = Tree(sample,existFeatures)

#xxx = 0
#for i in range(len(leaves)):
#    Entropy.append(getEntropy(leaves[i]))
#    xxx = xxx + len(leaves[i])

dropout_rate = []
for i in range(len(leaves)):
    leaf = leaves[i]
    dropout_rate.append((len(leaf)-sum(leaf[:,0])) / len(leaf[:,0]))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
