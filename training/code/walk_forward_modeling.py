"""
    File Name: walk_forward_modeling.py
    Date: 9/18/2019
    Updated:
    Author: reed.clarke@rcsoftwareservices.com
"""

import sys
import numpy as np
import pandas as pd
from reader.stock_reader import YahooFinanceDataReader


def get_tickers(tickers, start_date, end_date, absolute_or_relative_csv_file_path):
    reader = YahooFinanceDataReader('get_data', test=True)
    out_df = pd.DataFrame()
    for ticker in tickers:
        print("ticker = ", ticker)
        try:
            df = reader.get_stock_data(ticker, start_date, end_date,
                                       absolute_or_relative_csv_file_path,
                                       download=True, index_as_date=False, append_csv=True)
            df[['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']]
            df['mid'] = df['high'] + df['low'] / 2
            df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
            df['ticker'] = ticker
            df = df.set_index(['date', 'ticker'])
            out_df = pd.concat([out_df, df], axis=0)
        except TypeError as e:
            print("No data found for: ", ticker)
    return out_df.sort_index()


prices = get_tickers(['AAPL', 'CSCO', 'AMZN', 'MSFT'], '2012-01-01', end_date=None,
                     absolute_or_relative_csv_file_path="C:/Stock Manager/training/df_files/yahoo-finance")

# print(prices.info())
# print(prices.tail())

num_obs = prices.close.count()

def add_memory(s,n_days=50,memory_strength=0.1):
    ''' adds autoregressive behavior to series of data'''
    add_ewm = lambda x: (1-memory_strength)*x + memory_strength*x.ewm(n_days).mean()
    out = s.groupby(level='ticker').apply(add_ewm)
    return out

# generate feature data
f01 = pd.Series(np.random.randn(num_obs),index=prices.index)
f01 = add_memory(f01,10,0.1)
f02 = pd.Series(np.random.randn(num_obs),index=prices.index)
f02 = add_memory(f02,10,0.1)
f03 = pd.Series(np.random.randn(num_obs),index=prices.index)
f03 = add_memory(f03,10,0.1)
f04 = pd.Series(np.random.randn(num_obs),index=prices.index)
f04 = f04 # no memory

## now, create response variable such that it is related to features
# f01 becomes increasingly important, f02 becomes decreasingly important,
# f03 oscillates in importance, f04 is stationary, finally a noise component is added

outcome =   f01 * np.linspace(0.5,1.5,num_obs) + \
            f02 * np.linspace(1.5,0.5,num_obs) + \
            f03 * pd.Series(np.sin(2*np.pi*np.linspace(0,1,num_obs)*2)+1,index=f03.index) + \
            f04 + \
            np.random.randn(num_obs) * 3
outcome.name = 'outcome'

from sklearn.linear_model import LinearRegression
model = LinearRegression()
features = pd.concat([f01,f02,f03,f04],axis=1)

for index, row in features.iterrows():
    print(index)
    print(row)

sys.exit()

features.columns = ['f01','f02','f03','f04']
model.fit(X=features,y=outcome)
print('RSQ: '+str(model.score(X=features,y=outcome)))
print('Regression Coefficients: '+str(model.coef_))

split_point = int(0.80*len(outcome))

X_train = features.iloc[:split_point,:]
y_train = outcome.iloc[:split_point]
X_test = features.iloc[split_point:,:]
y_test = outcome.iloc[split_point:]

model = LinearRegression()
model.fit(X=X_train,y=y_train)

print('RSQ in sample: '+str(model.score(X=X_train,y=y_train)))
print('RSQ out of sample: '+str(model.score(X=X_test,y=y_test)))
print('Regression Coefficients: '+str(model.coef_))

recalc_dates = features.resample('Q', level='date').mean().index.values[:-1]
print('recalc_dates:')
print(recalc_dates)
print()

sys.exit()

models = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(None, date), level='date', drop_level=False)
    y_train = outcome.xs(slice(None, date), level='date', drop_level=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    models.loc[date] = model

    # print("Training on the first {} records, through {}" \
    #       .format(len(y_train), y_train.index.get_level_values('date').max()))
    # print("Coefficients: {}".format((model.coef_)))

def extract_coefs(models):
    coefs = pd.DataFrame()
    for i,model in enumerate(models):
        model_coefs = pd.Series(model.coef_,index=['f01','f02','f03','f04']) #extract coefficients for model
        model_coefs.name = models.index[i] # name it with the recalc date
        coefs = pd.concat([coefs, model_coefs], axis=1, sort=True)
    return coefs.T
extract_coefs(models).plot(title='Coefficients for Expanding Window Model')
plt.show()

recalc_dates = features.resample('Q', level='date').mean().index.values[:-1]

models = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(date - pd.Timedelta('90 days'), date), level='date', drop_level=False)
    y_train = outcome.xs(slice(date - pd.Timedelta('90 days'), date), level='date', drop_level=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    models.loc[date] = model

    # print("Training on the most recent {} records".format(len(y_train)))
    # print("Coefficients: {}".format((model.coef_)))

# extract_coefs(models).plot(title='Coefficients for Rolling Window Model')
# plt.show()

begin_dates = models.index
end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))

predictions = pd.Series(index=features.index)

for i,model in enumerate(models): #loop thru each models object in collection
    X = features.xs(slice(begin_dates[i],end_dates[i]),level='date',drop_level=False)
    p = pd.Series(model.predict(X),index=X.index)
    predictions.loc[X.index] = p

print(predictions.shape)

models_expanding_window = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(None, date), level='date', drop_level=False)
    y_train = outcome.xs(slice(None, date), level='date', drop_level=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    models_expanding_window.loc[date] = model

models_rolling_window = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(date - pd.Timedelta('90 days'), date), level='date', drop_level=False)
    y_train = outcome.xs(slice(date - pd.Timedelta('90 days'), date), level='date', drop_level=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    models_rolling_window.loc[date] = model

begin_dates = models.index
end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))

predictions_expanding_window = pd.Series(index=features.index)
for i, model in enumerate(models_expanding_window):  # loop thru each models object in collection
    X = features.xs(slice(begin_dates[i], end_dates[i]), level='date', drop_level=False)
    p = pd.Series(model.predict(X), index=X.index)
    predictions_expanding_window.loc[X.index] = p

predictions_rolling_window = pd.Series(index=features.index)
for i, model in enumerate(models_rolling_window):  # loop thru each models object in collection
    X = features.xs(slice(begin_dates[i], end_dates[i]), level='date', drop_level=False)
    p = pd.Series(model.predict(X), index=X.index)
    predictions_rolling_window.loc[X.index] = p

from sklearn.metrics import r2_score

common_idx = outcome.dropna().index.intersection(predictions_expanding_window.dropna().index)
rsq_expanding = r2_score(y_true = outcome[common_idx],y_pred=predictions_expanding_window[common_idx])
rsq_rolling = r2_score(y_true = outcome[common_idx],y_pred=predictions_rolling_window[common_idx])

print("Expanding Window RSQ: {}".format(round(rsq_expanding,3)))
print("Rolling Window RSQ: {}".format(round(rsq_rolling,3)))

from sklearn.tree import DecisionTreeRegressor

split_point = int(0.80*len(outcome))

X_train = features.iloc[:split_point,:]
y_train = outcome.iloc[:split_point]
X_test = features.iloc[split_point:,:]
y_test = outcome.iloc[split_point:]

model = DecisionTreeRegressor(max_depth=3)
model.fit(X=X_train,y=y_train)

print('RSQ in sample: '+str(round(model.score(X=X_train,y=y_train),3)))
print('RSQ out of sample: '+str(round(model.score(X=X_test,y=y_test),3)))

recalc_dates = features.resample('Q', level='date').mean().index.values[:-1]

models_rolling_window = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(date - pd.Timedelta('365 days'), date), level='date', drop_level=False)
    y_train = outcome.xs(slice(date - pd.Timedelta('365 days'), date), level='date', drop_level=False)
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X_train, y_train)
    models_rolling_window.loc[date] = model

predictions_rolling_window = pd.Series(index=features.index)
for i, model in enumerate(models_rolling_window):  # loop thru each models object in collection
    X = features.xs(slice(begin_dates[i], end_dates[i]), level='date', drop_level=False)
    p = pd.Series(model.predict(X), index=X.index)
    predictions_rolling_window.loc[X.index] = p

common_idx = y_test.dropna().index.intersection(predictions_rolling_window.dropna().index)
rsq_rolling = r2_score(y_true=y_test[common_idx], y_pred=predictions_rolling_window[common_idx])
print("RSQ out of sample (rolling): {}".format(round(rsq_rolling, 3)))

sys.exit()
