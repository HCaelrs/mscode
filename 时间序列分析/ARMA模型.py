# encoding:utf-8
from statsmodels.tsa.arima_model import ARMA
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ads.csv')
data = df.Ads
# data = df
n = 10
order = (6, 3)
train = data[:-n]
test = data[-n:]
lth = len(train)

tempModel = ARMA(train, order).fit()
delta = tempModel.fittedvalues - train  # 残差
score = 1 - delta.var() / train.var()
predicts = tempModel.predict(lth, lth+n-1, dynamic=True)
print(len(predicts))
print("predicts:\n", predicts, "\noriginal\n", test)


comp = pd.DataFrame()
comp['original'] = test
comp['predict'] = predicts
comp.plot()
plt.show()
