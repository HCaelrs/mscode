from statsmodels.tsa.stattools import adfuller
import pandas as pd
"""
ADF：float
    测试统计。
pvalue：float
    probability value：MacKinnon基于MacKinnon的近似p值（1994年，2010年）。
usedlag：int
    使用的滞后数量。
NOBS：int
    用于ADF回归和计算临界值的观察数。
critical values：dict
    测试统计数据的临界值为1％，5％和10％。基于MacKinnon（2010）。
icbest：float
    如果autolag不是None，则最大化信息标准。
"""

df = pd.read_csv("ads.csv")

print(adfuller(df.Ads))

