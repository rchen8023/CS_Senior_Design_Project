
import numpy as np

def node_feature(nodes,n,existFeatures,dn,feature_start,feature_end,threshold):
    entropy = float("inf")
    feature_index = -1
    
    bigEntropy = False
    leavesEntropies = []
    for i in range(dn):
        leavesEntropies.append(getEntropy(nodes[i]))
        if leavesEntropies[i] > threshold:
            bigEntropy = True
    
    if bigEntropy == False:
        return -1
    
    for index in range(feature_start,feature_end):
        if index not in existFeatures:
        
            entropy_total = 0
            
            for i in range(dn):
    #            if getEntropy(nodes[i]) > threshold:
                if leavesEntropies[i] > threshold:
                    
    #                if index not in existFeatures:
                    size = len(nodes[i])
                    left_temp,right_temp = split(index,nodes[i])
                    
                    num_left = len(left_temp)
                    num_right = len(right_temp)
                        
                    prob_feature_right = num_right/len(nodes[i])
                    prob_feature_left = num_left/len(nodes[i])
                    
                    right_entropy = getEntropy(right_temp)
                    left_entropy = getEntropy(left_temp)
                    
                    weighted_entropy = prob_feature_right*right_entropy + prob_feature_left*left_entropy
                    
                    weight = size/n
                    entropy_total = entropy_total + weight*weighted_entropy
                        
                        
    #                else:
    #                    entropy_total = entropy
    #                    break
    
            if entropy_total < entropy:
                entropy = entropy_total
                feature_index = index

    if feature_index != -1:    
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

def BuildTree(node,n,max_depth,existFeatures,feature_start,feature_end,threshold):
    Tree = [node]

    
    for i in range(max_depth):
        print('depth: ',i)
        dn = 2**i 
        if i == max_depth-1:
            break
        
        current_leaves = Tree[dn-1:]
        index = node_feature(current_leaves,n,existFeatures,dn,feature_start,feature_end,threshold)
        
        if index == -1:
            break
        
        for j in range(dn):
            
            if getEntropy(current_leaves[j]) <= threshold:
                left = current_leaves[j]
                right = []
            else:
                left,right = split(index,current_leaves[j])

            
            Tree.append(left)
            Tree.append(right)
        
    return Tree,dn

def Detect(sample,max_depth,threshold,feature_start,feature_end):

    [n,p] = sample.shape
    
    
    
    Features = []

    existFeatures = list()
#    np.random.shuffle(sample)
    
    
    Tree,dn = BuildTree(sample,n,max_depth,existFeatures,feature_start,feature_end,threshold)
    
    leaves = Tree[dn-1:]
    leaves_clean = [e for e in leaves if len(e) != 0]
    Features = existFeatures
        
    
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
            
    dropout_rate_clean = []
    entropy_clean = []
    for i in range(len(leaves_clean)):
        leaf_clean = leaves_clean[i]
        if len(leaf_clean)==0:
            dropout_rate_clean.append(-1)
            entropy_clean.append(-1)
        else:
            dropout_rate_clean.append((len(leaf_clean)-sum(leaf_clean[:,0])) / len(leaf_clean))
            entropy_clean.append(getEntropy(leaf_clean))
            
    return Features

    
#data = np.genfromtxt('data\crimecommunity.csv',delimiter=',',usecols=np.arange(0,102))
##data = np.genfromtxt('data\student-por_Cleaned.csv',delimiter=',',usecols=np.arange(0,46))
#data = np.nan_to_num(data)
#sample = data[:,:]
#[n,p] = sample.shape
#label = sample[:,0]
#feature_start = 1
#feature_end = 102
##max_depth = feature_end - feature_start
#max_depth = 31
#threshold = 0.3
#
#print('detecting...')
#Feature = Detect(sample,max_depth,threshold,feature_start,feature_end)
#print('done detecting')
    
    
    
    
    
    
    
    
    
    
