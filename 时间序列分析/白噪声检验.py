# encoding:utf-8
"""
acorr_ljungbox(x, lags=None, boxpierce=False)   # 数据的纯随机性检验函数
lags为延迟期数，如果为整数，则是包含在内的延迟期数，如果是一个列表或数组，那么所有时滞都包含在列表中最大的时滞中
boxpierce为True时表示除开返回LB统计量还会返回Box和Pierce的Q统计量
返回值：
lbvalue:测试的统计量
pvalue:基于卡方分布的p统计量
bpvalue:((optionsal), float or array) – 基于 Box-Pierce 的检验的p统计量
bppvalue:((optional), float or array) – 基于卡方分布下的Box-Pierce检验的p统计量
"""

from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# df = pd.read_csv("..\TestData\datasets\\austa.csv")
# InPt = list(df.value)
# 上述数据基本是随着时间的增加而增加 下面尝试一下所有随机的数据


# InPt = np.random.random(100)
# # 很显然数字随机的情况下得到的都是很大的p


df = pd.read_csv("ads.csv")
InPt = list(df.Ads)


# df = pd.read_csv("currency.csv")
# InPt = list(df.GEMS_GEMS_SPENT)

# result = acorr_ljungbox(InPt, lags=[6,12], boxpierce=True, return_df=True)
result = acorr_ljungbox(InPt, lags=20, boxpierce=True, return_df=True)
print(result)


plt.plot(df.Time, df.Ads)

plt.show()
