from utils import prepare_data, make_loader
import pandas as pd
import numpy as np
import time
from easydict import EasyDict

from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Model: k-Nearest Neighbour with n=20

# Load data
try:
    X_train, y_train, X_val, y_val, X_test, y_test= prepare_data()
except:
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    
    # y labels
    whole_y_train = pd.read_csv("data/Training/Annotation_Training.csv",
                                usecols = [i for i in range(1,64)],
                                skiprows = [1,2,3])
    whole_y_test = pd.read_csv("data/Testing/Annotation_Testing.csv",
                          usecols = [i for i in range(1,64)],
                          skiprows = [1,2,3])

    whole_y_train.set_axis([i for i in range(63)], axis=1, inplace=True)
    whole_y_test.set_axis([i for i in range(63)], axis=1, inplace=True)

    y_train = whole_y_train[:3000]
    y_test = whole_y_test[:300]

    # X_train = X_train[:40]
    # X_test = X_test[:40]
    # y_train = y_train[:40]
    # y_test = y_test[:40]

# Conversion
X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)

# Parameters
args = EasyDict({
    'n_neighbors':20
})

best_mse = 9999999999

print('k-Nearest Neighbour')
print("parameters:", args["n_neighbors"])
model = KNeighborsRegressor(**args)
model = MultiOutputRegressor(model)

start = time.time()
# Training
model.fit(X_train, y_train)

# Test
output = model.predict(X_test)

end = time.time()
Time = end-start

mse = mean_squared_error(y_test, output)

print('time:', Time, end='')
print(' mse:', mse)