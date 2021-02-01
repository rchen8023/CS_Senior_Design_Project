"""Implementation of the random decision tree algorithm."""
import numpy as np
from multiprocessing import Process, Queue

class Node:
    """Class implements a Node data structure."""
    def __init__(self, prob_0_class, prob_1_class, feature_index):
        self.prob_0_class = prob_0_class
        self.prob_1_class = prob_1_class
        self.feature_index = feature_index
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    """Class implements a binary classifier random decision tree."""
    
    def __init__(self, min_ranges, max_ranges, max_depth = None, min_samples_split = 2):
        """Sets max depth for the tree and initializes the tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_ranges = min_ranges
        self.max_ranges = max_ranges
        self.tree = None

    def fit(self, queue, X, y, valid_features):
        """Build a random decision tree classifier from the training set (X, y).
        Adds this tree to the queue object."""
        self.tree = self._grow_tree(X, y, valid_features)
        queue.put(self)

    def predict(self, X):
        """Predict binary class for input samples X"""
        return [self._predict(inputs) for inputs in X]

    def _grow_tree(self, X, y, valid_features, depth = 0):
        """Grows the depth of the tree by creating a random split on a node."""
        prob_0_class = sum(i==0 for i in y)/y.size
        prob_1_class = sum(i==1 for i in y)/y.size
        
        # Choose a random feature to split node at
        feature_index = np.random.choice(valid_features)
        
        # Remove chosen feature from possible features to be chosen later
        valid_features = np.delete(valid_features, np.argwhere(valid_features == feature_index))
        
        node = Node(prob_0_class = prob_0_class, prob_1_class = prob_1_class, feature_index = feature_index)
        
        # Continue if we are not at the max depth and there are features left to test
        if depth < self.max_depth and valid_features.size > 1:
            threshold = np.random.uniform(low = self.min_ranges[feature_index], high = self.max_ranges[feature_index])
            indices_left = X[:, feature_index] <= threshold
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            
            node.threshold = threshold
            
            # Only continue if there is data to generate both a left and right subtree
            # This will prune the nodes which have no data
            if X_left.size >= self.min_samples_split and X_right.size >= self.min_samples_split:
                node.left = self._grow_tree(X_left, y_left, valid_features, depth + 1)
                node.right = self._grow_tree(X_right, y_right, valid_features, depth + 1)
            
        return node

    def _predict(self, X):
        """Predict binary class for input samples X"""
        node = self.tree
        while node.left and node.right:
            if X[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prob_0_class, node.prob_1_class
    
class RandomForestClassifier:
    """Class implements a random forest binary classifier based on random decision trees."""
    
    def __init__(self, max_depth=None, min_samples_split = 1, num_trees=10):
        """Sets max depth, number of random trees and initializes the forest."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_trees = num_trees
        self.decision_trees = []
        
    def fit(self, X, y):
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
            bootstrap = np.random.choice(np.arange(X[:,0].size), size = int(1 * X[:,0].size), replace = False)  
            
            # Generate a feature subset
            features = np.random.choice(np.arange(X[0].size), size = int(1 * X[0].size), replace = False)

            q = Queue()
            p = Process(target=tree.fit, args=(q, X[bootstrap,:], y[bootstrap], features))
            processes.append(p)
            queues.append(q)
            p.start()
        
        # Add the constructed decision trees to a list
        for i in range(0, self.num_trees):
            out = queues[i].get()
            self.decision_trees[i] = out
            processes[i].join()

    def predict(self, X):
        """Predict binary class for input samples X"""
        pred_arr = []
        for dtree in self.decision_trees:
            pred_arr.append(dtree.predict(X))

        ave_arr = np.mean(pred_arr, axis=0)
        predictions = [np.argmax(i) for i in ave_arr]
        
        return predictions

