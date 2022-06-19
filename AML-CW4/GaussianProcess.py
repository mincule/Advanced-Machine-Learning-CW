from utils import prepare_data, make_loader
import pandas as pd
import numpy as np
import time
from easydict import EasyDict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, ConstantKernel, RBF
from sklearn.metrics import mean_squared_error

# Model: Gaussian Process with DotProduct+Constant Kernel, alpha=1e+5

# Load data
X_train, y_train, X_val, y_val, X_test, y_test= prepare_data()

# Conversion
X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)

# Parameters
args = EasyDict({
    "kernel" :DotProduct()+ConstantKernel(),
    "alpha" : 1e+5,
    "random_state": 0
})

best_mse = 9999999999

print('Gaussian Process Regressor')
print("parameters:", args["kernel"], args["alpha"])
model = GaussianProcessRegressor(**args)

start = time.time()
# Training
model.fit(X_train, y_train)

end = time.time()
Time = end-start

# Test
output = model.predict(X_test)
mse = mean_squared_error(y_test, output)

print('time:', Time, end='')
print(' mse:', mse)