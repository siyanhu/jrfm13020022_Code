import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from pmdarima import  auto_arima
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Loaddataset
path = "/Users/husiyan/Google Drive/备份-完成的课题与项目/研究-VirusPaper/data/version4/nv.csv"
names = ['Date', 'Num_confirmed_patients', 'Num_deaths']
dataset = pd.read_csv(path, names=names)

# Split trainning and precition
train_cls = {
    'Num_confirmed_deaths': dataset.iloc[:,1].values
}
train_dataset = pd.DataFrame(train_cls, columns = ['Num_confirmed_deaths'])
train_dataset.set_index(dataset.iloc[:, 0].values, inplace=True)
train_data = pd.Series(train_dataset['Num_confirmed_deaths'])
train_data.head()
train_data.plot(figsize=(20,10))
# plt.show()

#平稳性检验
result = ADF(train_dataset)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
plot_acf(train_data)
plot_pacf(train_data)
# plt.show()
(p, q) =(sm.tsa.arma_order_select_ic(train_data,max_ar=3 ,max_ma=3 ,ic='aic')['aic_min_order'])
print(p,q)

data = np.array(train_dataset.values).T[0]
fittedmodel = auto_arima(train_data, start_p=1, start_q=1, max_p=3, max_q=3, max_d=3,max_order=None,
                         seasonal=False, m=1, test='adf', trace=False,
                         error_action='ignore',  # don't want to know if an order does not work
                         suppress_warnings=True,  # don't want convergence warnings
                         stepwise=True, information_criterion='bic', njob=-1)  # set to stepwise
print(fittedmodel.summary())

def plot_arima(truth, forecasts, title="ARIMA", xaxis_label='Time',
               yaxis_label='Value', c1='#A6CEE3', c2='#B2DF8A',
               forecast_start=None, **kwargs):
    # make truth and forecasts into pandas series
    n_truth = truth.shape[0]
    n_forecasts = forecasts.shape[0]

    # always plot truth the same
    truth = pd.Series(truth, index=np.arange(truth.shape[0]))

    # if no defined forecast start, start at the end
    if forecast_start is None:
        idx = np.arange(n_truth, n_truth + n_forecasts)
    else:
        idx = np.arange(forecast_start, n_forecasts)
    forecasts = pd.Series(forecasts, index=idx)
    return forecasts


y_hat = np.append(fittedmodel.predict_in_sample(), fittedmodel.predict(20))
predict_dataset = plot_arima(data, y_hat,
                title="Original Series & In-sample Predictions",
                c2='#FF0000', forecast_start=0)
print(predict_dataset)
# predict_cls.plot(figsize=(20,10))
# plt.show()
