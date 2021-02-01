import numpy as np
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import RidgeClassifier

class DistributedFairRandomForest:
    """Class implements a distributed fair random forest binary classifier based on random decision trees."""
    
    def __init__(self, prot_class, max_depth=20, min_samples_leaf = 3, num_fair_trees = 50, alpha = 100, rho = .05):
        """Initializes the forest."""
        self.prot_class = prot_class
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.num_fair_trees = num_fair_trees
        self.alpha = alpha
        self.rho = rho
        self.decision_trees = []
        self.ridge_model = RidgeClassifier(alpha=alpha, fit_intercept = True)
        self.n_classes = 2
        self.classes_ = [0,1]
        
    def fit(self, X, y):
        """Build a random decision tree based classifier from the training set (X, y)."""
        
        # Remove protected features
        X_protect = np.delete(X, [self.prot_class], axis=1)
        
        num_tr = len(y)
        num_prot_1 = sum(X[:, self.prot_class])
        num_prot_0 = num_tr - num_prot_1
        
        #X_protect = X
        i = 0
        fair_trees = []
        predictions = []
        
        # Pick up fair trees
        while i < self.num_fair_trees:
            new_tree = ExtraTreeClassifier(max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, max_features = 1)
            new_tree.fit(X_protect, y)
            new_prediction = new_tree.predict(X_protect)
            
            # Calculate the probability we predict someone will dropout between groups (Statistical Parity)
            num_pred_1 = len([e for e in range(0, num_tr) if new_prediction[e] == 0 and X[e,self.prot_class] == 1])
            num_pred_0 = len([e for e in range(0, num_tr) if new_prediction[e] == 0 and X[e,self.prot_class] == 0])
            stat_parity = abs(num_pred_1/num_prot_1 - num_pred_0/num_prot_0)
            
            if stat_parity < self.rho:
                i += 1
                fair_trees.append(new_tree)
                predictions.append(new_prediction)
            
        self.ridge_model.fit(np.transpose(np.asarray(predictions)), y)
        self.decision_trees = fair_trees
        
    def predict(self, X):
        """Predict binary class for input samples X"""
        pred_arr = []
        
        X_protect = np.delete(X, [self.prot_class], axis=1)
        
        pred_arr = [dtree.predict(X_protect) for dtree in self.decision_trees]
        
        predictions = self.ridge_model.predict(np.transpose(np.asarray(pred_arr)))
        pred = [round(i) for i in predictions]

        return pred
