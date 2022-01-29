"""
    File Name: lstm_mulitifeature.py
    Date: 3/8/2020
    Updated:
    Author: reed.clarke@rcsoftwareservices.com
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv("../code/csv_files/yahoo-finance/AAL-get-historical-data (2000-01-01 to 2020-01-01).csv")
print(dataset_train.info())
training_set = dataset_train.iloc[:, [6, 4]].values
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train, y_train)
