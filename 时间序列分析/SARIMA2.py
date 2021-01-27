# encoding:utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

plt.style.use('fivethirtyeight')
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 28, 18
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

df = pd.read_csv('ads.csv')

#
# for i in range(df.shape[0]):
#     df.iloc[i, 0] = df.iloc[i, 0].replace('T', ' ')


for i in range(df.shape[0]):
    df.iloc[i, 0] = i

data = df

data = data.Ads
decomposition = seasonal_decompose(data, freq=24, model="multiplicative")
trend = decomposition.trend  # 趋势效应
seasonal = decomposition.seasonal  # 季节效应
residual = decomposition.resid  # 随机效应
print(residual.__class__)
plt.subplot(411)
plt.plot(data, label=u'原始数据')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label=u'趋势')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label=u'季节性')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label=u'残差')
plt.legend(loc='best')
plt.show()

# (0, 1, 0)  (1, 1, 1, 12)
# 上面是真实跑出来的----hc
# 模型的建立
mod = sm.tsa.statespace.SARIMAX(data,
                                order=(0, 1, 0),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())

# 模型检验
# 模型诊断
results.plot_diagnostics(figsize=(15, 12))
plt.show()
# LB检验
r, q, p = sm.tsa.acf(results.resid.values.squeeze(), qstat=True)
data = np.c_[range(1, 41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
