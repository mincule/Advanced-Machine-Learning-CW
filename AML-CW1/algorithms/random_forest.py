# Reference
# python-engineer, MLfromscratch, (2021), GitHub repository,
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/random_forest.py

from collections import Counter
import numpy as np
from algorithms.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, max_depth, min_sample_split, max_features, n_trees=3):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        self.n_trees = n_trees
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                self.max_depth,
                self.min_sample_split,
                self.max_features
            )
            bootstrap_samples = self._bootstrap(X, y)
            tree.fit(*bootstrap_samples)
            self.trees.append(tree)
    
    def predict(self, X):
        y_pred = []
        tree_pred = np.array([tree.predict(X) for tree in self.trees])
        tree_pred = np.swapaxes(tree_pred, 1, 0)
        for feature in tree_pred:
            feature = Counter(feature)
            label = max(feature.keys(), key=lambda k : feature[k])
            y_pred.append(label)
        return np.array(y_pred)
    
    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]