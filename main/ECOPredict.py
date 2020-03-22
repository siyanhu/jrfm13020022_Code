import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from pmdarima import  auto_arima
import datetime
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# Loaddataset
path = "/Users/husiyan/Google Drive/备份-完成的课题与项目/研究-VirusPaper/review/data/seasonal.csv"
names = ['Quarter', 'GDP', 'EPIC']
dataset = pd.read_csv(path, names=names)

# Split trainning and precition
train_cls = {
    'GDP': dataset.iloc[:,1].values,
    'EPIC': dataset.iloc[:,2].values
}

train_dataset = pd.DataFrame(train_cls, columns = ['GDP'])
train_dataset.set_index(dataset.iloc[:, 0].values, inplace=True)
train_data = pd.Series(train_dataset['GDP'])
train_data.head()
# train_data.plot(figsize=(20,10))
# plt.show()

#平稳性检验
#0
result = ADF(train_dataset)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
plot_acf(train_data)
plot_pacf(train_data)
plt.show()
(p, q) =(sm.tsa.arma_order_select_ic(train_data,max_ar=3 ,max_ma=3 ,ic='aic')['aic_min_order'])
print(p,q)

#1
# diff1 = train_data.diff(1)
# diff1.dropna(inplace=True)
# result = ADF(diff1)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# plot_acf(diff1)
# plot_pacf(diff1)
# plt.show()
# (p, q) =(sm.tsa.arma_order_select_ic(diff1,max_ar=3 ,max_ma=3 ,ic='aic')['aic_min_order'])
# print(p,q)


#2
# diff2 = train_data.diff(2)
# diff2.dropna(inplace=True)
# result = ADF(diff2)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# plot_acf(diff2)
# plot_pacf(diff2)
# plt.show()
# (p, q) =(sm.tsa.arma_order_select_ic(diff2,max_ar=3 ,max_ma=2 ,ic='aic')['aic_min_order'])
# print(p,q)


arima110 = sm.tsa.SARIMAX(train_data, order=(3, 2, 0), trend='c')
res = arima110.fit()
print(res.summary())
print(res.forecast(4))