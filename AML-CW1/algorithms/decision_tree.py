from collections import Counter

import numpy as np


# Outer modules
def entropy(y):
    # y (List[int]): class labels
    _, count = np.unique(y, return_counts=True)
    ps = count / len(y)
    return -np.sum([p*np.log2(p) for p in ps])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

# Main module - Decision Tree
class DecisionTree:
    def __init__(self, max_depth, min_sample_split, max_features, criterion=entropy):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        self.criterion = criterion
        self.root = None
    
    def fit(self, X, y):
        # X (ndarray): image data (N by feature_size)
        # y (ndarray): labels (N)
        self.root = self._create_tree(X, y)
    
    def predict(self, X):
        if len(X.shape) == 1: # one datapoint case
            return self._traverse_tree(X, self.root)
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _create_tree(self, X, y, depth=0):
        """ Making tree!
        [STEP]
        0th - check the stopping criteria
        1st - determine the split point
        2nd - split and get child nodes
        3rd - make trees with child nodes
        """
        n_sample, feature_size = X.shape
        n_feature_type = len(np.unique(y))
        
        # 0th STEP - stopping criteria
        if (
            depth >= self.max_depth
            or n_sample < self.min_sample_split
            or n_feature_type == 1
        ):
            leaf = Counter(y)
            leaf_value = max(leaf.keys(), key=lambda k : leaf[k])
            return Node(value=leaf_value)
        
        # 1st STEP
        split_point = self._determine_split_point(X, y)
        # 2nd STEP
        _, _, left_child, right_child = self._split_tree(*split_point, X, y)
        # 3rd STEP
        left_node = self._create_tree(X[left_child], y[left_child], depth+1)
        right_node = self._create_tree(X[right_child], y[right_child], depth+1)
        return Node(*split_point, left_node, right_node)
    
    def _information_gain(self, split_idx, split_val, X, y):
        # Parent entropy
        entropy_parent = self.criterion(y)
        
        # Split and Get Child entropy
        left_y, right_y, _, _ = self._split_tree(split_idx, split_val, X, y)
        left_child, right_child = self.criterion(left_y), self.criterion(right_y)
        entropy_child = (len(left_y)/len(y))*left_child + (len(right_y)/len(y))*right_child
        return entropy_parent - entropy_child
    
    def _split_tree(self, split_idx, split_val, X, y):
        left_y, right_y = [], []
        left_point, right_point = [], []
        for idx, img in enumerate(X):
            if img[split_idx] < split_val:
                left_y.append(y[idx])
                left_point.append(idx)
            else:
                right_y.append(y[idx])
                right_point.append(idx)
        return left_y, right_y, left_point, right_point
    
    def _determine_split_point(self, X, y):
        # random feature selection
        idx_features = np.random.choice(len(X[0]), self.max_features, replace=False)

        best_gain = -1
        split_idx, split_val = None, None

        # Compare the info. gain
        for i in idx_features:
            val_feature = np.unique(X[:,i])
            for val in val_feature:
                gain = self._information_gain(i, val, X, y)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_val = val
        return split_idx, split_val

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)