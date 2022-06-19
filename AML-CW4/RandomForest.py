from utils import prepare_data, make_loader
import pandas as pd
import numpy as np
import time
from easydict import EasyDict

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Model: Random Forest max_depth=50, max_features=sqrt, min_samples_split=10, n_estimators=3

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

    # X_train = X_train[:4]
    # X_test = X_test[:4]
    # y_train = y_train[:4]
    # y_test = y_test[:4]

# Conversion
X_train = X_train.reshape(X_train.shape[0],-1)
X_test = X_test.reshape(X_test.shape[0],-1)

# Parameters
args = EasyDict({
"n_estimators": 3,
'max_depth': 50,
'min_samples_split': 10,
'max_features': "sqrt",
"random_state": 0
})

best_mse = 9999999999

print('Random Forest Regressor')
print("parameters:", args["max_depth"], args["max_features"], args["min_samples_split"], args["n_estimators"])
model = RandomForestRegressor(**args)

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