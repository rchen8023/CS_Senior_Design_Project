import numpy as np
from sklearn.tree import ExtraTreeClassifier

class FairExtraRandomForestClassifier:
    """Class implements a random forest binary classifier based on random decision trees."""
    
    def __init__(self, gender_fair, address_fair, max_depth=None, min_samples_split = 1, num_trees=10):
        """Sets max depth, number of random trees and initializes the forest."""
        self.gender_fair = gender_fair
        self.location_fair = address_fair
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_trees = num_trees
        self.decision_trees = []
        
    def fit(self, X, y, num_fair_trees = 60, max_depth = 23):
        """Build a random decision tree based classifier from the training set (X, y).
        Adds this tree to the queue object."""
        
        # for-loop to generate multiple base models 
        for i in range(0, self.num_trees):
            # train one base model 
            self.decision_trees.append(ExtraTreeClassifier(max_depth = max_depth, min_samples_leaf = 1, max_features = 1))
            
        # Fit the decision trees
        for tree in self.decision_trees:
            tree.fit(X, y)
            
        # Pick the 30 most fair trees
        dem_parity_gen = []
        dem_parity_add = []
        for i in range(0, self.num_trees):
            predictions = self.decision_trees[i].predict(X)
            
            num_tr = len(y)
            num_w = sum(X[:, 0])
            num_m = len(y) - num_w
            num_u = sum(X[:, 16])
            num_r = len(y) - num_u
            
            # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
            num_men_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,0] == 0])
            num_women_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,0] == 1])
            num_urban_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,16] == 1])
            num_rural_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,16] == 0])
        
            dem_parity_gen.append(abs(num_men_pred/num_m - num_women_pred/num_w))
            dem_parity_add.append(abs(num_urban_pred/num_u - num_rural_pred/num_r))
        
        sum_parity = [dem_parity_gen[i]+dem_parity_add[i] for i in range(len(dem_parity_gen))]
        
        if self.location_fair and self.gender_fair:
            indices = np.asarray(sum_parity).argsort()[:num_fair_trees]
        elif self.gender_fair:
            indices = np.asarray(dem_parity_gen).argsort()[:num_fair_trees]
        elif self.address_fair:
            indices = np.asarray(dem_parity_add).argsort()[:num_fair_trees]
        
        fair_trees = []
        for i in indices:
            fair_trees.append(self.decision_trees[i])
            
        self.decision_trees = fair_trees
        
    def predict(self, X):
        """Predict binary class for input samples X"""
        pred_arr = []
        
        for dtree in self.decision_trees:
            pred_arr.append(dtree.predict(X))

        ave_arr = np.mean(pred_arr, axis=0)
        predictions = [round(i) for i in ave_arr]
        
        return predictions
    