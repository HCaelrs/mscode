# encoding:utf-8
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ads.csv")
inpt = df.Ads
print('自相关系数: \n', stattools.acf(inpt))
print('偏自相关系数: \n', stattools.pacf(inpt))

inpt = inpt.dropna()

plot_acf(inpt, lags=100)
plt.show()

plot_pacf(inpt, lags=100)
plt.show()

# 可以看到，这里自相关系数和偏自相关系数的计算都返回了长度为10的数组。它们分别代表了当l ll的取值为从0到10时的自相关性统计
