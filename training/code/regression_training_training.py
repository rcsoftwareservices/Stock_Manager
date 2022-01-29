"""
    Name: regression_training_training.py
    Date: 8 / 8 / 2019
    Author: reed.clarke @ rcsoftwareservices.com
"""

from reader.stock_reader import QuandlDataReader
import math
import numpy as np
import pandas as pd
import pickle

# from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.ticker as mtick
from matplotlib import style

register_matplotlib_converters()

style.use('ggplot')

# df = quandl.get("WIKI/GOOGL")
dr = QuandlDataReader('../../csv_files/quandl/')
df = dr.get_data_from_csv_file('GOOGL')
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')
df = dr.fill_in_missing_dates(df)
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
# "Label" - parameter being predicted.
# "Feature" - parameters used to predict "Label" values.
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_validate = X[-forecast_out:]
X = X[:-forecast_out]
df = df.dropna()
y = np.array(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle','wb') as file:
#     pickle.dump(clf, file)
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)
confidence = clf.score(X_test, y_test)
forecast_set = clf.predict(X_validate)
print(forecast_set, confidence, forecast_out)
date_values = df.index.values
df['Forecast'] = np.nan
min_date = date_values[0]
max_date = date_values[-1]
for i in forecast_set:
    max_date += np.timedelta64(1, 'D')
    df.loc[max_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick)
df = df[-2000:]
df['Adj. Close'].plot(label='Adj. Close')
df['Forecast'].plot(label='Forecast Price')
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
# ax.xaxis.set_major_locator(mdates.MonthLocator())
# fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('GOOGL - LinearRegression')
ax.legend(loc='lower right')
# plt.grid()
plt.show()