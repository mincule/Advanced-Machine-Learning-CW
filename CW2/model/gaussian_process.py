import numpy as np
from scipy.linalg import kron
from scipy.spatial.distance import cdist
    
# utils
def vec(v):
    """
    [[a,a'],      [[a],
     [b,b'],  ->   [b],
     [c,c']]       [c],
                   [a'],
                   [b'],
                   [c']]
    """
    return np.reshape(v.T, (-1,1))

def unvec(v, r, c):
    return np.reshape(v, (c,r)).T


# Exact Multi-output Gaussian Process
class MOGPR:
    """
    Input
    -----
    Xtrain: (N_train, D)
    ytrain: (N_train, D)
    kernel: Hyperparameter
    coregionalisation_matrix: Identity matrix
    sigma: Hyperparameter
    
    Output
    ------
    mean: (N_test, D)
    """
    def __init__(self, Xtrain, ytrain, kernel, coregionalisation_matrix="I",  sigma=1.0):
        self.X_train = np.array(Xtrain)
        self.y_train = np.array(ytrain)
        self.kernel = kernel
        self.C = coregionalisation_matrix
        self.sigma = sigma
        self._predictive_distribution = None
    
    def predict(self, Xtest):
        Xtest = np.array(Xtest)
        N, D = self.y_train.shape
        if self.C == "I":
            self.C = np.identity(D)
        
        Sigma = np.diag(np.full(D,self.sigma))
        
        # Kernel matrices
        K = self.kernel(self.X_train, self.X_train) # (N_train, N_train)
        K_star = self.kernel(Xtest, self.X_train) # (N_test, N_train)
        K_star_2 = self.kernel(Xtest, Xtest) # (N_test, N_test)
        
        # Block matrices
        CK = kron(self.C, K)
        CK_star = kron(self.C, K_star)
        CK_starT = kron(self.C, K_star.T)
        CK_star_2 = kron(self.C, K_star_2)
        Sigma_block = kron(Sigma, np.identity(N))
        
        # Predictive distribution
        Common = CK_star @ np.linalg.inv(CK + Sigma_block)
        mean = Common @ vec(self.y_train)
        cov = CK_star_2 - Common @ CK_starT
        var = np.diag(cov)
        
        mean = unvec(mean, D, Xtest.shape[0]).T
        self._predictive_distribution = {"mean": mean, "cov": cov, "var": var}
        return mean

# Subset of Regressors approximation of Multi-output Gaussian Process
class SR_MOGPR:
    """
    From training set, it produces inducing inputs (Zx, Zy) and get an approximate predictive dist.
    
    Input
    -----
    Xtrain: (N_train, D)
    ytrain: (N_train, D)
    kernel: Hyperparameter
    B: (float) Ratio of inducing inputs to Xtrain, Hyperparameter
    coregionalisation_matrix: Identity matrix
    sigma: Hyperparameter
    
    Output
    ------
    mean: (N_test, D)
    """
    def __init__(self, Xtrain, ytrain, kernel, B=0.1, coregionalisation_matrix="I",  sigma=1.0):
        self.X_train = np.array(Xtrain)
        self.y_train = np.array(ytrain)
        self.kernel = kernel
        self.B = B
        self.C = coregionalisation_matrix
        self.sigma = sigma
        self._predictive_distribution = None
        
        self.Zx = None
        self.Zy = None
        
    def predict(self, Xtest):
        Xtest = np.array(Xtest)
        N, D = self.y_train.shape
        if self.C == "I":
            self.C = np.identity(D)
        
        Sigma = np.diag(np.full(D,self.sigma**2))
        
        # Select inducing inputs - M = N_train*B
        idx = np.array(np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0]*self.B), replace=False))
        self.Zx = np.array(self.X_train[idx])
        self.Zy = np.array(self.y_train[idx])
        
        # Approximate kernel matrices
        K_bb = self.kernel(self.Zx, self.Zx) # (M, M)
        K_xb = self.kernel(self.X_train, self.Zx) # (N_train, M)
        K_starb = self.kernel(Xtest, self.Zx) # (N_test, M)
        K_star_2 = self.kernel(Xtest, Xtest) # (N_test, N_test)
        
        KtK = K_xb.T @ K_xb # (M, M)
        
        # Block matrices
        CK_starb = kron(self.C, K_starb) # (D*N_test, DM)
        CK_bstar = kron(self.C, K_starb.T) # (DM, D*N_test)
        CKtK = kron(self.C, KtK) # (DM, DM)
        Sigma_block = kron(Sigma, K_bb)
        inverse_block = np.linalg.inv(CKtK + Sigma_block)
        
        # Predictive distribution
        mean = inverse_block @ vec(K_xb.T @ self.y_train) # (DM, 1)
        mean = CK_starb @ mean # (D*N_test,1)
        cov = (self.sigma**2) * CK_starb @ inverse_block @ CK_bstar
        var = np.diag(cov)
        
        mean = unvec(mean, D, Xtest.shape[0]).T
        self._predictive_distribution = {"mean": mean, "cov": cov, "var": var}
        return mean