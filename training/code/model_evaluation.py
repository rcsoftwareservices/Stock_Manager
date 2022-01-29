"""
    File Name: model_evaluation.py
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
    out_df = out_df.sort_index()
    return out_df


prices = get_tickers(['AAPL', 'CSCO', 'AMZN', 'MSFT'], '2012-01-01', end_date=None,
                     absolute_or_relative_csv_file_path="C:/Stock Manager/training/df_files/yahoo-finance")

num_obs = prices.close.count()


def add_memory(s,n_days=50,mem_strength=0.1):
    ''' adds autoregressive behavior to series of data'''
    add_ewm = lambda x: (1-mem_strength)*x + mem_strength*x.ewm(n_days).mean()
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

features = pd.concat([f01,f02,f03,f04],axis=1)

## now, create response variable such that it is related to features
# f01 becomes increasingly important, f02 becomes decreasingly important,
# f03 oscillates in importance, f04 is stationary,
# and finally a noise component is added

outcome =   f01 * np.linspace(0.5,1.5,num_obs) + \
            f02 * np.linspace(1.5,0.5,num_obs) + \
            f03 * pd.Series(np.sin(2*np.pi*np.linspace(0,1,num_obs)*2)+1,index=f03.index) + \
            f04 + \
            np.random.randn(num_obs) * 3
outcome.name = 'outcome'

from sklearn.linear_model import LinearRegression

## fit models for each timestep on a walk-forward basis
recalc_dates = features.resample('Q',level='date').mean().index.values[:-1]
models = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.xs(slice(None,date),level='date',drop_level=False)
    y_train = outcome.xs(slice(None,date),level='date',drop_level=False)
    model = LinearRegression()
    model.fit(X_train,y_train)
    models.loc[date] = model

## predict values walk-forward (all predictions out of sample)
begin_dates = models.index
end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))

predictions = pd.Series(index=features.index)

for i,model in enumerate(models): #loop thru each models object in collection
    X = features.xs(slice(begin_dates[i],end_dates[i]),level='date',drop_level=False)
    p = pd.Series(model.predict(X),index=X.index)
    predictions.loc[X.index] = p

import sklearn.metrics as metrics

# make sure we have 1-for-1 mapping between pred and true
common_idx = outcome.dropna().index.intersection(predictions.dropna().index)
y_true = outcome[common_idx]
y_true.name = 'y_true'
y_pred = predictions[common_idx]
y_pred.name = 'y_pred'

standard_metrics = pd.Series()

standard_metrics.loc['explained variance'] = metrics.explained_variance_score(y_true, y_pred)
standard_metrics.loc['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
standard_metrics.loc['MSE'] = metrics.mean_squared_error(y_true, y_pred)
standard_metrics.loc['MedAE'] = metrics.median_absolute_error(y_true, y_pred)
standard_metrics.loc['RSQ'] = metrics.r2_score(y_true, y_pred)

print(standard_metrics)

print(pd.concat([y_pred, y_true], axis=1).tail())

def make_df(y_pred, y_true):
    y_pred.name = 'y_pred'
    y_true.name = 'y_true'
    df = pd.concat([y_pred, y_true], axis=1)
    df['sign_pred'] = df.y_pred.apply(np.sign)
    df['sign_true'] = df.y_true.apply(np.sign)
    df['is_correct'] = 0
    df.loc[
        df.sign_pred * df.sign_true > 0, 'is_correct'] = 1  # only registers 1 when prediction was made AND it was correct
    df['is_incorrect'] = 0
    df.loc[
        df.sign_pred * df.sign_true < 0, 'is_incorrect'] = 1  # only registers 1 when prediction was made AND it was wrong
    df['is_predicted'] = df.is_correct + df.is_incorrect
    df['result'] = df.sign_pred * df.y_true
    return df


df = make_df(y_pred, y_true)
print(df.dropna().tail())

"""
Accuracy: 
    Just as the name suggests, this measures the percent of predictions that were directionally
    correct vs. incorrect.
Edge: 
    Perhaps the most useful of all metric-limits, 
    this is the expected value of the prediction over a sufficiently large set of draws. 
    Think of this like a blackjack card counter who knows the expected profit on each dollar bet when the odds are at a
    level of favorability.
Noise: 
    Critically important but often ignored, 
    the noise metric estimates how dramatically the model's predictions vary from one day to the next. 
    As you might i+9++++magine, a model which abruptly changes its mind every few days is much harder to follow 
    (and much more expensive to trade) than one which is a bit more steady.
    
y_true_chg and y_pred_chg: 
    The average magnitude of change (per period) in y_true and y_pred.
prediction_calibration: 
    A simple ratio of the magnitude of our predictions vs. magnitude of truth. 
    This gives some indication of whether our model is properly tuned to the size of movement in addition to the direction
    of it.
capture_ratio: 
    Ratio of the "edge" we gain by following our predictions vs. the actual daily change. 
    100 would indicate that we were perfectly capturing the true movement of the target variable.
edge_long and edge_short: 
    The "edge" for only long signals or for short signals.
edge_win and edge_lose: 
    The "edge" for only winners or for only losers.    
    
"""


def calc_scorecard(df):
    scorecard = pd.Series()
    # building block metric-limits
    scorecard.loc['accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    # derived metric-limits
    scorecard.loc['y_true_chg'] = df.y_true.abs().mean()
    scorecard.loc['y_pred_chg'] = df.y_pred.abs().mean()
    scorecard.loc['prediction_calibration'] = scorecard.loc['y_pred_chg']/scorecard.loc['y_true_chg']
    scorecard.loc['capture_ratio'] = scorecard.loc['edge']/scorecard.loc['y_true_chg']*100
    # metric-limits for a subset of predictions
    scorecard.loc['edge_long'] = df[df.sign_pred == 1].result.mean() - df.y_true.mean()
    scorecard.loc['edge_short'] = df[df.sign_pred == -1].result.mean() - df.y_true.mean()
    scorecard.loc['edge_win'] = df[df.is_correct == 1].result.mean() - df.y_true.mean()
    scorecard.loc['edge_lose'] = df[df.is_incorrect == 1].result.mean() - df.y_true.mean()
    return scorecard

print(calc_scorecard(df))

"""
    1. The model is predicting with a strong directional accuracy
    2. We are generating about 1.4 units of "edge" (expected profit) each prediction, 
        which is about half of the total theoretical profit
    3. The model makes more on winners than it loses on losers
    4. The model is equally valid on both long and short predictions
"""

"""
    Critically important when considering using a model in live trading is to understand:
        (a) how consistent the model's performance has been, and 
        (b) whether its current performance has degraded from its past. 
        Markets have a way of discovering and eliminating past sources of edge.
"""

def scorecard_by_year(df):
    df['year'] = df.index.get_level_values('date').year
    return df.groupby('year').apply(calc_scorecard).T

print(scorecard_by_year(df))

"""
    Compare performance across tickers.
"""

def scorecard_by_symbol(df):
    return df.groupby(level='ticker').apply(calc_scorecard).T

print(scorecard_by_symbol(df))

"""
Comparing models:
    The added insight we get from this methodology comes when wanting to make comparisons between 
    models, periods, segments, etc...
    To illustrate, let's say that we're comparing two models, a linear regression vs. a random forest, 
    for performance on a training set and a dev set 
    (pretend for a moment that we didn't adhere to Walk-forward model building practices...).
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

X_train,X_test,y_train,y_test = train_test_split(features,outcome,test_size=0.20,shuffle=False)

# linear regression
model1 = LinearRegression().fit(X_train,y_train)
model1_train = pd.Series(model1.predict(X_train),index=X_train.index)
model1_test = pd.Series(model1.predict(X_test),index=X_test.index)

model2 = SVR(kernel='rbf', C=500, gamma='scale').fit(X_train, y_train)
# model2 = RandomForestRegressor().fit(X_train,y_train)
model2_train = pd.Series(model2.predict(X_train),index=X_train.index)
model2_test = pd.Series(model2.predict(X_test),index=X_test.index)

# create dataframes for each
model1_train_df = make_df(model1_train,y_train)
model1_test_df = make_df(model1_test,y_test)
model2_train_df = make_df(model2_train,y_train)
model2_test_df = make_df(model2_test,y_test)

s1 = calc_scorecard(model1_train_df)
s1.name = 'model1_train'
s2 = calc_scorecard(model1_test_df)
s2.name = 'model1_test'
s3 = calc_scorecard(model2_train_df)
s3.name = 'model2_train'
s4 = calc_scorecard(model2_test_df)
s4.name = 'model2_test'

print(pd.concat([s1,s2,s3,s4],axis=1))

sys.exit()
