"""
    File Name: ml_stock_prediction.py
    Date: 9/9/2019
    Updated:
    Author: reed.clarke@rcsoftwareservices.com
"""

# TODO - CHECK mid. price as a feature selection.

import sys
import inspect
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from reader.stock_reader import YahooFinanceDataReader


def get_stock_portfolio_data(tickers, start_date, end_date=None, absolute_or_relative_csv_file_path=None):
    print(inspect.currentframe().f_code.co_name)
    dr = YahooFinanceDataReader('get_data')
    portfolio_df = pd.DataFrame()
    for ticker in tickers:
        print("ticker = ", ticker)
        ticker_df = dr.get_stock_data(ticker, start_date, end_date,
                                      absolute_or_relative_csv_file_path,
                                      download=True, index_as_date=False, append_csv=True)
        portfolio_df = format_stock_portfolio_data(ticker, ticker_df, portfolio_df)
    return portfolio_df.sort_index()


def format_stock_portfolio_data(ticker, ticker_df, portfolio_df):
    print(inspect.currentframe().f_code.co_name)
    ticker_df[['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']]
    # ticker_df = ticker_df.rename(columns={'adjclose': 'adj.close'})
    ticker_df['mid'] = ticker_df['high'] + ticker_df['low'] / 2
    # tick_df['^DJI']
    ticker_df['date'] = pd.to_datetime(ticker_df['date'], format="%Y-%m-%d")
    ticker_df['ticker'] = ticker
    ticker_df = ticker_df.set_index(['date', 'ticker'])
    return pd.concat([portfolio_df, ticker_df], axis=0)


def get_stock_portfolio_features(portfolio_df):
    print(inspect.currentframe().f_code.co_name)
    features_df = pd.DataFrame(index=portfolio_df.index)
    features_df['adjclose_chg'] = portfolio_df.groupby(level='ticker').adjclose.pct_change(1)
    features_df['mid_chg'] = portfolio_df.groupby(level='ticker').mid.pct_change(1)
    features_df['volume_change_ratio'] = portfolio_df.groupby(level='ticker').volume.diff(1) / \
                                         portfolio_df.groupby(level='ticker').shift(1).volume
    features_df['momentum_5_day'] = portfolio_df.groupby(level='ticker').close.pct_change(5)
    features_df['intraday_chg'] = (portfolio_df.groupby(level='ticker').close.shift(0) -
                                   portfolio_df.groupby(level='ticker').open.shift(0)) / \
                                   portfolio_df.groupby(level='ticker').open.shift(0)
    # features_df['day_of_week'] = features_df.index.get_level_values('date').weekday
    # features_df['day_of_month'] = features_df.index.get_level_values('date').day
    features_df = features_df.dropna()
    return features_df


def get_stock_portfolio_outcomes(portfolio_df):
    print(inspect.currentframe().f_code.co_name)
    outcomes_df = pd.DataFrame(index=portfolio_df.index)
    # next day's opening change
    outcomes_df['open_1'] = portfolio_df.groupby(level='ticker').open.shift(-1) \
                            / portfolio_df.groupby(level='ticker').close.shift(0) - 1
    # next day's closing change
    func_one_day_ahead = lambda x: x.pct_change(-1)
    outcomes_df['close_1'] = portfolio_df.groupby(level='ticker').close \
        .apply(func_one_day_ahead)
    func_five_day_ahead = lambda x: x.pct_change(-5)
    outcomes_df['close_5'] = portfolio_df.groupby(level='ticker').close \
        .apply(func_five_day_ahead)
    print((outcomes_df.tail(25)))
    return outcomes_df


def calculate_stock_feature_ranking(features_df, outcomes_df):
    print(inspect.currentframe().f_code.co_name)
    model = RandomForestRegressor(max_features=3)
    y = outcomes_df.open_1
    X = features_df
    Xy = X.join(y).dropna()
    y = Xy[y.name]
    X = Xy[X.columns]
    print(y.shape)
    print(X.shape)
    model.fit(X, y)
    print("Model Score: " + str(model.score(X, y)))
    print("Feature Importance: ")
    print(pd.Series(model.feature_importances_, index=X.columns) \
          .sort_values(ascending=False))


tickers = ['AAPL']
start_date = '2000-01-01'
absolute_or_relative_csv_file_path = "C:/Stock Manager/dev/df_files/yahoo-finance"
stock_portfolio_df = get_stock_portfolio_data(tickers, start_date, end_date=None,
                                              absolute_or_relative_csv_file_path=absolute_or_relative_csv_file_path)

print(stock_portfolio_df.info())
print(stock_portfolio_df.head())

stock_portfolio_features_df = get_stock_portfolio_features(stock_portfolio_df)
print(stock_portfolio_features_df.info())
print(stock_portfolio_features_df.head())

stock_portfolio_outcomes_df = get_stock_portfolio_outcomes(stock_portfolio_df)
calculate_stock_feature_ranking(stock_portfolio_features_df, stock_portfolio_outcomes_df)

sys.exit()
