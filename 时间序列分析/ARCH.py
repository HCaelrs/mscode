# encoding:utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import arch
from scipy import linalg

warnings.filterwarnings("ignore")

df = pd.read_csv('ads.csv')

#
# for i in range(df.shape[0]):
#     df.iloc[i, 0] = df.iloc[i, 0].replace('T', ' ')


for i in range(df.shape[0]):
    df.iloc[i, 0] = i

data = df

data = data.Ads
mod = sm.tsa.statespace.SARIMAX(data,
                                order=(0, 1, 0),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary())
print((results.fittedvalues - data))

at = (results.fittedvalues - data)[1:]

at.plot()
plt.show()

at2 = np.square(at)
plt.show()

m = 25  # 我们检验25个自相关系数
acf, q, p = sm.tsa.acf(at2, nlags=m, qstat=True)  ## 计算自相关系数 及p-value
out = np.c_[range(1, 26), acf[1:], q, p]
output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
output = output.set_index('lag')

print(output)

fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(111)
sm.graphics.tsa.plot_pacf(at2, lags=30, ax=ax1)
plt.show()

print(data)
am = arch.arch_model(data, mean='AR', lags=8, vol='ARCH', p=4)
res = am.fit()
print(res.summary())

res.plot()
plt.plot(data)
plt.show()

res.hedgehog_plot()
plt.show()

