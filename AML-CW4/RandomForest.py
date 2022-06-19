from utils import prepare_data, make_loader
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import time
from easydict import EasyDict

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import time

# Model: Random Forest max_depth=50, max_features=sqrt, min_samples_split=10, n_estimators=3

# Load data
X_train, y_train, X_val, y_val, X_test, y_test= prepare_data()

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