"""
    File Name: ensemble_model_stock_prediction.py
    Date: 9/19/2019
    Updated:
    Author: reed.clarke@rcsoftwareservices.com
"""

import sys
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from reader.stock_reader import YahooFinanceDataReader

register_matplotlib_converters()


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


#  NOTE; only using Yahoo Finance prices to generate realistic multi-index dates and tickers
idx = get_tickers(['AAPL', 'CSCO', 'MSFT', 'INTC'], '2012-01-01', end_date=None,
                  absolute_or_relative_csv_file_path="C:/Stock Manager/training/df_files/yahoo-finance").index

num_obs = len(idx)
split = int(num_obs * .80)

## First, create factors hidden within feature set
hidden_factor_1 = pd.Series(np.random.randn(num_obs), index=idx)
hidden_factor_2 = pd.Series(np.random.randn(num_obs), index=idx)
hidden_factor_3 = pd.Series(np.random.randn(num_obs), index=idx)
hidden_factor_4 = pd.Series(np.random.randn(num_obs), index=idx)

## Next, generate outcome variable outcomes that is related to these hidden factors
outcomes = (0.5 * hidden_factor_1 + 0.5 * hidden_factor_2 +  # factors linearly related to outcome
            hidden_factor_3 * np.sign(hidden_factor_4) + hidden_factor_4 * np.sign(
            hidden_factor_3) +  # factors with non-linear relationships
            pd.Series(np.random.randn(num_obs), index=idx)).rename('outcomes')  # noise

print(type(outcomes))

## Generate features which contain a mix of one or more hidden factors plus noise and bias

f1 = 0.25 * hidden_factor_1 + pd.Series(np.random.randn(num_obs), index=idx) + 0.5
f2 = 0.5 * hidden_factor_1 + pd.Series(np.random.randn(num_obs), index=idx) - 0.5
f3 = 0.25 * hidden_factor_2 + pd.Series(np.random.randn(num_obs), index=idx) + 2.0
f4 = 0.5 * hidden_factor_2 + pd.Series(np.random.randn(num_obs), index=idx) - 2.0
f5 = 0.25 * hidden_factor_1 + 0.25 * hidden_factor_2 + pd.Series(np.random.randn(num_obs), index=idx)
f6 = 0.25 * hidden_factor_3 + pd.Series(np.random.randn(num_obs), index=idx) + 0.5
f7 = 0.5 * hidden_factor_3 + pd.Series(np.random.randn(num_obs), index=idx) - 0.5
f8 = 0.25 * hidden_factor_4 + pd.Series(np.random.randn(num_obs), index=idx) + 2.0
f9 = 0.5 * hidden_factor_4 + pd.Series(np.random.randn(num_obs), index=idx) - 2.0
f10 = hidden_factor_3 + hidden_factor_4 + pd.Series(np.random.randn(num_obs), index=idx)

## From these features, create an features_df

features_df = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], axis=1)
features_df.columns = features_df.columns = ['f01', 'f02', 'f03', 'f04', 'f05', 'f06', 'f07', 'f08', 'f09', 'f10']

print(type(features_df))

## Distribution of features_df and target
# features_df.plot.kde(legend=True, xlim=(-5, 5), color=['green'] * 5 + ['orange'] * 5,
#                      title='Distributions - Features and Target')
# outcomes.plot.kde(legend=True, linestyle='--', color='red')  # target
# plt.show()
#
# sns.set(style="dark")
#
# # Set up the matplotlib figure
# fig, axes = plt.subplots(4, 3, figsize=(8, 6), sharex=True, sharey=True)
#
# # Rotate the starting point around the cubehelix hue circle
# for ax, s in zip(axes.flat, range(10)):
#     cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)
#     x = features_df.iloc[:, s]
#     sns.regplot(x, outcomes, fit_reg=True, marker=',', scatter_kws={'s': 1}, ax=ax, color='salmon')
#     ax.set(xlim=(-5, 5), ylim=(-5, 5))
#     ax.text(x=0, y=0, s=x.name.upper(), color='black',
#             **{'ha': 'center', 'va': 'center', 'family': 'sans-serif'}, fontsize=20)
#
# fig.tight_layout()
# fig.suptitle("Univariate Regressions for Features", y=1.05, fontsize=20)
# plt.show()

# from scipy.cluster import hierarchy
# from scipy.spatial import distance
#
# corr_matrix = features_df.corr()
# correlations_array = np.asarray(corr_matrix)
# linkage = hierarchy.linkage(distance.pdist(correlations_array), \
#                             method='average')
# g = sns.clustermap(corr_matrix, row_linkage=linkage, col_linkage=linkage, \
#                    row_cluster=True, col_cluster=True, figsize=(5, 5), cmap='Greens', center=0.5)
# label_order = corr_matrix.iloc[:, g.dendrogram_row.reordered_ind].columns
# plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
# plt.show()


def make_walkforward_model(features_df, outcomes, algo=LinearRegression()):
    recalc_dates = features_df.resample('Q', level='date').mean().index.values[:-1]
    ## Train models
    models = pd.Series(index=recalc_dates)
    for date in recalc_dates:
        X_train = features_df.xs(slice(None, date), level='date', drop_level=False)
        y_train = outcomes.xs(slice(None, date), level='date', drop_level=False)
        # print(f'Train with data prior to: {date} ({y_train.count()} obs)')
        model = clone(algo)
        model.fit(X_train, y_train)
        models.loc[date] = model
    begin_dates = models.index
    end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))
    ## Generate OUT OF SAMPLE walk-forward predictions
    predictions = pd.Series(index=features_df.index)
    for i, model in enumerate(models):  # loop thru each models object in collection
        # print(f'Using model trained on {begin_dates[i]}, Predict from: {begin_dates[i]} to: {end_dates[i]}')
        x = features_df.xs(slice(begin_dates[i], end_dates[i]), level='date', drop_level=False)
        p = pd.Series(model.predict(x), index=x.index)
        predictions.loc[x.index] = p
    return models, predictions


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

linear_models, linear_preds = make_walkforward_model(features_df, outcomes, algo=LinearRegression())
tree_models, tree_preds = make_walkforward_model(features_df, outcomes, algo=ExtraTreesRegressor(n_estimators=10))

# print("Models:")
# print("\tLinear:")
# print(linear_models.head())
# print("\tTree:")
# print(tree_models.head())
# print("Predictions:")
# print("\tLinear:")
# print(linear_preds.dropna().head())
# print("\tTree:")
# print(tree_preds.dropna().head())

# pd.DataFrame([model.coef_ for model in linear_models],
#              columns=features_df.columns,index=linear_models.index).\
#     plot(title='Weighting Coefficients for \nLinear Model')
# plt.show()

from sklearn.metrics import r2_score, mean_absolute_error


def calc_scorecard(y_pred, y_true):
    def make_df(y_pred, y_true):
        y_pred.name = 'y_pred'
        y_true.name = 'y_true'
        df = pd.concat([y_pred, y_true], axis=1).dropna()
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
    scorecard = pd.Series()
    # building block metric-limits
    scorecard.loc['RSQ'] = r2_score(df.y_true, df.y_pred)
    scorecard.loc['MAE'] = mean_absolute_error(df.y_true, df.y_pred)
    scorecard.loc['directional_accuracy'] = df.is_correct.sum() * 1. / (df.is_predicted.sum() * 1.) * 100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    # derived metric-limits
    scorecard.loc['edge_to_noise'] = scorecard.loc['edge'] / scorecard.loc['noise']
    scorecard.loc['edge_to_mae'] = scorecard.loc['edge'] / scorecard.loc['MAE']
    scorecard.loc['edge_to_mae_plus_edge'] = scorecard.loc['edge'] / (scorecard.loc['MAE'] + scorecard.loc['edge'])
    return scorecard


# calc_scorecard(y_pred=linear_preds, y_true=outcomes).rename('Linear')


def scores_over_time(y_pred,y_true):
    df = pd.concat([y_pred,y_true],axis=1).dropna().reset_index().set_index('date')
    scores = df.resample('A').apply(lambda df: calc_scorecard(df[y_pred.name],df[y_true.name]))
    return scores


# scores_by_year = scores_over_time(y_pred=linear_preds,y_true=outcomes)
# print(scores_by_year.tail(3).T)
# scores_by_year['edge_to_mae'].plot(title='Prediction Edge vs. MAE')
# plt.show()

# Make Ensemble Model
from sklearn.linear_model import LassoCV
def prepare_Xy(X_raw,y_raw):
    """ Utility function to drop any samples without both valid X and y values"""
    Xy = X_raw.join(y_raw).replace({np.inf:None,-np.inf:None}).dropna()
    X = Xy.iloc[:,:-1]
    y = Xy.iloc[:,-1]
    return X,y
X_ens, y_ens = prepare_Xy(X_raw=pd.concat([linear_preds.rename('linear'),tree_preds.rename('tree')],
                                          axis=1), y_raw=outcomes)

ensemble_models,ensemble_preds = make_walkforward_model(X_ens,y_ens,algo=LassoCV(cv=5, positive=True))
ensemble_preds = ensemble_preds.rename('ensemble')
ensemble_preds.dropna()

# print(ensemble_preds.dropna().head())
#
# pd.DataFrame([model.coef_ for model in ensemble_models],
#              columns=X_ens.columns,index=ensemble_models.index).\
#     plot(title='Weighting Coefficients for \nSimple Two-Model Ensemble')
# plt.show()

# calculate scores for each model
# score_ens = calc_scorecard(y_pred=ensemble_preds, y_true=y_ens).rename('Ensemble')
# score_linear = calc_scorecard(y_pred=linear_preds, y_true=y_ens).rename('Linear')
# score_tree = calc_scorecard(y_pred=tree_preds, y_true=y_ens).rename('Tree')
# scores = pd.concat([score_linear,score_tree,score_ens],axis=1)
# scores.loc['edge_to_noise'].plot.bar(color='grey',legend=True)
# scores.loc['edge'].plot(color='green',legend=True)
# scores.loc['noise'].plot(color='red',legend=True)
# plt.show()
# print(scores)

fig,[[ax1,ax2],[ax3,ax4]] = plt.subplots(2,2,figsize=(9,6))
metric = 'RSQ'
scores_over_time(y_pred=ensemble_preds.rename('ensemble'),y_true=outcomes)[metric].rename('Ensemble').\
plot(title=f'{metric.upper()} over time',legend=True,ax=ax1)
scores_over_time(y_pred=linear_preds.rename('linear'),y_true=outcomes)[metric].rename('Linear').\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax1)
scores_over_time(y_pred=tree_preds.rename('tree'),y_true=outcomes)[metric].rename("Tree").\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax1)

metric = 'edge'
scores_over_time(y_pred=ensemble_preds.rename('ensemble'),y_true=outcomes)[metric].rename('Ensemble').\
plot(title=f'{metric.upper()} over time',legend=True,ax=ax2)
scores_over_time(y_pred=linear_preds.rename('linear'),y_true=outcomes)[metric].rename('Linear').\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax2)
scores_over_time(y_pred=tree_preds.rename('tree'),y_true=outcomes)[metric].rename("Tree").\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax2)

metric = 'noise'
scores_over_time(y_pred=ensemble_preds.rename('ensemble'),y_true=outcomes)[metric].rename('Ensemble').\
plot(title=f'{metric.upper()} over time',legend=True,ax=ax3)
scores_over_time(y_pred=linear_preds.rename('linear'),y_true=outcomes)[metric].rename('Linear').\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax3)
scores_over_time(y_pred=tree_preds.rename('tree'),y_true=outcomes)[metric].rename("Tree").\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax3)
metric = 'edge_to_mae_plus_edge'
scores_over_time(y_pred=ensemble_preds.rename('ensemble'),y_true=outcomes)[metric].rename('Ensemble').\
plot(title=f'{metric.upper()} over time',legend=True,ax=ax4)
scores_over_time(y_pred=linear_preds.rename('linear'),y_true=outcomes)[metric].rename('Linear').\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax4)
scores_over_time(y_pred=tree_preds.rename('tree'),y_true=outcomes)[metric].rename("Tree").\
plot(title=f'{metric.upper()} over time',legend=True, alpha = 0.5, linestyle='--',ax=ax4)

plt.tight_layout()
plt.show()


"""
Observations:
    1.  We can see that the ensemble is fairly consistently more effective than either of the base models.
    2.  All models seem to be getting better over time (and as they have more data on which to train).
    3.  The ensemble also appears to be a bit more consistent over time. 
        Much like a diversified portfolio of stocks should be less volatile than the individual stocks within it,
        an ensemble of diverse models will often perform more consistently across time.

Next Steps:
    1.  More model types: add SVMs, deep learning models, regularlized regressions, and dimensionality reduction models
        to the mix.
    2.  More hyperparameter combinations: try multiple sets of hyperparameters on a particular algorithm.
    3.  Orthogonal feature sets: try training base models on different subsets of features. 
        Avoid the "curse of dimensionality" by limiting each base model to an appropriately small number of features.
"""

sys.exit()
