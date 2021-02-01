"""Implementation of the random decision tree algorithm."""
import numpy as np
from multiprocessing import Process, Queue

class Node:
    """Class implements a Node data structure."""
    def __init__(self, prob_0_class, prob_1_class, feature_index):
        self.prob_0_class = prob_0_class
        self.prob_1_class = prob_1_class
        self.feature_index = feature_index
        self.threshold = []
        self.children = []

class DecisionTreeClassifier:
    """Class implements a binary classifier random decision tree."""
    
    def __init__(self, min_ranges, max_ranges, max_depth = None, min_samples_split = 1):
        """Sets max depth for the tree and initializes the tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_ranges = min_ranges
        self.max_ranges = max_ranges
        self.tree = None

    def fit(self, queue, X, y, valid_features, categorical):
        """Build a random decision tree classifier from the training set (X, y).
        Adds this tree to the queue object."""
        self.tree = self._grow_tree(X, y, valid_features, categorical)
        queue.put(self)

    def predict(self, X):
        """Predict binary class for input samples X"""
        return [self._predict(inputs) for inputs in X]

    def _grow_tree(self, X, y, valid_features, categorical, depth = 0):
        """Grows the depth of the tree by creating a random split on a node."""
        if y.size != 0:
            prob_0_class = sum(i==0 for i in y)/y.size
            prob_1_class = sum(i==1 for i in y)/y.size
        else:
            prob_0_class = 0
            prob_1_class = 0
            
        # Choose a random feature to split node at
        feature_index = np.random.choice(valid_features)
        
        # Remove chosen feature from possible features to be chosen later
        valid_features = np.delete(valid_features, np.argwhere(valid_features == feature_index))
        
        node = Node(prob_0_class = prob_0_class, prob_1_class = prob_1_class, feature_index = feature_index)
        
        # Continue if we are not at the max depth and there are features left to test
        if depth < self.max_depth and valid_features.size >= 1:
            threshold = []
            
            # If the feature is continuous
            if categorical[feature_index] == 0:
                threshold.append(np.random.uniform(low = self.min_ranges[feature_index], high = self.max_ranges[feature_index]))
                indices_left = X[:, feature_index] <= threshold[0]
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                
                # Only continue if there is data to generate both a left and right subtree
                # This will prune the nodes which have no data
                if X_left.size >= self.min_samples_split and X_right.size >= self.min_samples_split:
                    node.threshold = threshold
                    node.children.append(self._grow_tree(X_left, y_left, valid_features, categorical, depth + 1))
                    node.children.append(self._grow_tree(X_right, y_right, valid_features, categorical, depth + 1))
            
            # The feature is categorical
            else:
                valid_data = True
                X_data = []
                y_data = []
                
                for i in range(0, categorical[feature_index]):
                    threshold.append(i)
                    indices = X[:, feature_index] == threshold[i]
                    X_data.append(X[indices])
                    y_data.append(y[indices])
                    if X_data[i].size < self.min_samples_split:
                        valid_data = False
                        break
                
                # If there is data in each child node
                if valid_data:
                    node.threshold = threshold
                    # For each category in this feature
                    for i in range(0, categorical[feature_index]):
                        node.children.append(self._grow_tree(X_data[i], y_data[i], valid_features, categorical, depth + 1))
            
        return node

    def _predict(self, X):
        """Predict binary class for input samples X"""
        node = self.tree
        while len(node.children) != 0:
            if len(node.children) == 2:
                if X[node.feature_index] <= node.threshold[0]:
                    node = node.children[0]
                else:
                    node = node.children[1]
                
            else:
                for i in range(0, len(node.children)):
                    if X[node.feature_index] <= node.threshold[i]:
                        node = node.children[i]
                        break
                    
                break

        return node.prob_0_class, node.prob_1_class
    
class FairGenderRandomForestClassifier:
    """Class implements a random forest binary classifier based on random decision trees."""
    
    def __init__(self, max_depth=None, min_samples_split = 1, num_trees=10):
        """Sets max depth, number of random trees and initializes the forest."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_trees = num_trees
        self.decision_trees = []
        
    def fit(self, X, y, categorical, num_fair_trees = 30, bootstrap_percent = 1, bagging_percent = 1):
        """Build a random decision tree based classifier from the training set (X, y).
        Adds this tree to the queue object."""
        min_ranges = [min(X[:,i]) for i in range(0, len(X[0]))]
        max_ranges = [max(X[:,i]) for i in range(0, len(X[0]))]
        
        self.decision_trees = []
        for i in range(0, self.num_trees):
            self.decision_trees.append(DecisionTreeClassifier(min_ranges, max_ranges, max_depth=self.max_depth, min_samples_split = self.min_samples_split))
        
        processes = []
        queues = []
        
        # Fit the decision trees in parallel
        for tree in self.decision_trees:
            # Generate a bootstrap sample
            bootstrap = np.random.choice(np.arange(X[:,0].size), size = int(bootstrap_percent * X[:,0].size), replace = False)  
            
            # Generate a feature subset
            features = np.random.choice(np.arange(X[0].size), size = int(bagging_percent * X[0].size), replace = False)

            q = Queue()
            p = Process(target=tree.fit, args=(q, X[bootstrap,:], y[bootstrap], features, categorical))
            processes.append(p)
            queues.append(q)
            p.start()
        
        # Add the constructed decision trees to a list
        for i in range(0, self.num_trees):
            out = queues[i].get()
            self.decision_trees[i] = out
            processes[i].join()
            
        # Pick the 30 most fair trees
        dem_parity_gen = []
        dem_parity_add = []
        for i in range(0, self.num_trees):
            pred_arr = self.decision_trees[i].predict(X)
            predictions = [np.argmax(i) for i in pred_arr]
            
            num_tr = len(y)
            num_w = sum(X[:, 0])
            num_m = len(y) - num_w
            num_u = sum(X[:, 1])
            num_r = len(y) - num_u
            
            # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
            num_men_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,0] == 0])
            num_women_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,0] == 1])
            num_urban_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,1] == 1])
            num_rural_pred = len([e for e in range(0, num_tr) if predictions[e] == 0 and X[e,1] == 0])
        
            dem_parity_gen.append(abs(num_men_pred/num_m - num_women_pred/num_w))
            dem_parity_add.append(abs(num_urban_pred/num_u - num_rural_pred/num_r))
        
        indices = np.asarray(dem_parity_gen).argsort()[:num_fair_trees]
        
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
        predictions = [np.argmax(i) for i in ave_arr]
        
        return predictions