from distributedFairRandomForest import DistributedFairRandomForest

def dfrfProcess(q, num_models, max_depth, min_samples_leaf, i, rho, alpha, prot_class, sample_train, sample_train_removed, label_train):
    dfrf = DistributedFairRandomForest(prot_class = prot_class, max_depth = max_depth, min_samples_leaf = min_samples_leaf, num_fair_trees = num_models, alpha = alpha, rho = rho)
    
    # Fit the model to our training set
    dfrf.fit(sample_train,label_train)
    
    q.put(i)
    q.put(dfrf)
