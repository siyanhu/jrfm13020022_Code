import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from pmdarima import auto_arima

import warnings
warnings.filterwarnings('ignore', 'statsmodels', FutureWarning)

# Loaddataset
path = "/Users/husiyan/Google Drive/备份-完成的课题与项目/研究-VirusPaper/review/data/seasonal.csv"
names = ['Quarter', 'GDP', 'EPIC']
dataset = pd.read_csv(path, names=names)

# Split trainning and precition
df = pd.DataFrame(dataset)
train_dataset = df[:][['GDP']]
train_dataset.set_index(dataset.iloc[:, 0].values, inplace=True)
train_data = pd.Series(train_dataset['GDP'])
train_data.head()

# path = "/Users/husiyan/Google Drive/备份-完成的课题与项目/研究-VirusPaper/review/result_gdp/output/"
# text_file = open(path, "w")
# n = text_file.write(predict_dataset.to_string())
# # text_file.close()


# auto regression
# out = 'AIC: {0:0.3f}, HQIC: {1:0.3f}, BIC: {2:0.3f}'
# ar = sm.tsa.AutoReg(train_dataset, lags=4, seasonal=True)
# res = ar.fit()
# print(res.summary())
# predictor=res.predict(len(train_data), len(train_data) + 3)
# print(predictor)
# predictor.plot(figsize=(20,10))
# plt.show()

# Arma
# (p, q) = (sm.tsa.arma_order_select_ic(train_data,max_ar=3 ,max_ma=3 ,ic='aic')['aic_min_order'])
# arma = sm.tsa.ARMA(train_dataset, order=(p,q))
# res = arma.fit()
# print(res.summary())
# print(res.predict(4))
# The computed initial AR coefficients are not stationary
# You should induce stationarity, choose a different model order, or you can
# pass your own start_params.

# Arima
# (p, q) = (sm.tsa.arma_order_select_ic(train_data, max_ar=3 ,max_ma=3,ic='aic')['aic_min_order'])
# d = 3
# diff_data = train_data.diff(d)
# diff_data.dropna(inplace=True)
# arima = sm.tsa.ARIMA(diff_data, order=(p,d,q))
# res = arima.fit()
# print(res.summary())

# SARIMAX: Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors
# def plot_arima(truth, forecasts, title="ARIMA", xaxis_label='Time',
#                yaxis_label='Value', c1='#A6CEE3', c2='#B2DF8A',
#                forecast_start=None, **kwargs):
#     # make truth and forecasts into pandas series
#     n_truth = truth.shape[0]
#     n_forecasts = forecasts.shape[0]
#
#     # always plot truth the same
#     truth = pd.Series(truth, index=np.arange(truth.shape[0]))
#
#     # if no defined forecast start, start at the end
#     if forecast_start is None:
#         idx = np.arange(n_truth, n_truth + n_forecasts)
#     else:
#         idx = np.arange(forecast_start, n_forecasts)
#     forecasts = pd.Series(forecasts, index=idx)
#     return forecasts
#
# sarimax = auto_arima(train_data, start_p=1, start_q=1, max_p=3, max_q=3, max_d=3,max_order=None,
#                          seasonal=False, m=1, test='adf', trace=False,
#                          error_action='ignore',  # don't want to know if an order does not work
#                          suppress_warnings=True,  # don't want convergence warnings
#                          stepwise=True, information_criterion='bic', njob=-1)  # set to stepwise
# print(sarimax.summary())
# predictor = sarimax.predict(4)
# print(predictor)

# Simple Exponential Smoothing (SES)
# ses = sm.tsa.SimpleExpSmoothing(train_data)
# res = ses.fit()
# print(res.summary())
# predictor=res.predict(len(train_data), len(train_data) + 3)
# print(predictor)
# predictor.plot(figsize=(20,10))
# plt.show()

# Holt Winter’s Exponential Smoothing (HWES)
hwes = sm.tsa.ExponentialSmoothing(train_data, seasonal_periods=4, seasonal='multiplicative')
res = hwes.fit()
print(res.summary())
predictor=res.predict(len(train_data), len(train_data) + 3)
print(predictor)
# predictor.plot(figsize=(20,10))
# plt.show()

