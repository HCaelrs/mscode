# encoding:utf-8
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import warnings
# import arch
# from scipy import linalg
#
# df = pd.read_csv('ads.csv')
#
# data = df
# data = data.Ads
# data = np.asarray(data)
#
# md = sm.tsa.VAR(data)
# re = md.fit()
# fevd = re.fevd(10)
# # 打印出方差分解的结果
# print(fevd.summary())
# # 画图
# fevd.plot(figsize=(12, 16))
# plt.show()

# some example data
import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

mdata = sm.datasets.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]
from statsmodels.tsa.base.datetools import dates_from_str
# 注意需要将索引设置为时间

quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp', 'realcons', 'realinv']]
mdata.index = pandas.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()

# make a VAR model
model = VAR(data)
results = model.fit(2)
results.summary()
