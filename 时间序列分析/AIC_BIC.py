import pandas as pd

import statsmodels.api as sm

df = pd.read_csv('daily-minimum-temperatures.csv', header=0)
data = df.Temp

ans = sm.tsa.arma_order_select_ic(data, max_ar=6, max_ma=4, ic='aic')
print(ans)
