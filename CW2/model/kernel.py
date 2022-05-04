import numpy as np
from scipy.spatial.distance import cdist

# Kernels
class LinearKernel:
    """
    Input
    -----
    X1: (N1, D)
    X2: (N2, D)
    
    Output
    ------
    kernal matrix: (N1, N2)
    
    """
    def __init__(self, sigma_0=1.0):
        self.sigma_0 = sigma_0
    
    def __call__(self, X1, X2):
        return self.sigma_0 + X1 @ X2.T

class GaussianKernel:
    """
    Input
    -----
    X1: (N1, D)
    X2: (N2, D)
    
    Output
    ------
    kernal matrix: (N1, N2)
    """
    def __init__(self, sigma_k=1.0):
        self.sigma_k = sigma_k
    
    def __call__(self, X1, X2):
        dist = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-dist/self.sigma_k**2)