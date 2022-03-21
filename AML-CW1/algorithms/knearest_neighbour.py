import numpy as np
from collections import Counter

def distance_L1(x1, x2):
    return np.sum(np.abs(x1-x2))

def distance_L2(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNearestNeighbour:
    def __init__(self, k, criterion=distance_L1):
        self.k = k
        self.criterion = criterion
        
    def fit(self, X, y):
        self.trainX = X
        self.trainy = np.array(y)
    
    def predict(self, X):
        y_pred = [self._predict_one(a) for a in X]
        return np.array(y_pred)
    
    def _predict_one(self, x):
        distance = [self.criterion(x,b) for b in self.trainX]
        idx = np.argsort(distance)[:self.k]
        k_labels = self.trainy[idx]
        k_labels = Counter(k_labels)
        label = max(k_labels.keys(), key=lambda k : k_labels[k])
        return label