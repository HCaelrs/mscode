# -*- coding: utf-8 -*-
# @Time : 2020/3/3 10:10
# @Author : lhy

# 引入相关的统计包
import warnings  # 忽略警告
import numpy as np  # 矢量和矩阵
import pandas as pd  # 表格和数据操作
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt


from itertools import product
from tqdm import tqdm_notebook
import statsmodels.api as sm



# 误差值 将其中每一个元素相减 除以真值就是误差 误差的平均值乘100 就是其误差率
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()



warnings.filterwarnings('ignore')

# =====================================================================================

# 1 如真实的手机游戏数据，将调查每小时观看的广告和每天花费的游戏币
ads = pd.read_csv(r'./ads.csv', index_col=['Time'], parse_dates=['Time'])
# =====================================================================================

# 建模 SARIMA
# setting initial values and some bounds for them
ps = range(2, 5)
d = 1
qs = range(2, 5)
Ps = range(0, 2)
D = 1
Qs = range(0, 2)
s = 24  # season length

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
print(parameters)
print(parameters_list)
print(len(parameters_list))


# -------------------------------------------------------------------------------------
def optimizeSARIMA(parameters_list, d, D, s):
    """
    Return dataframe with parameters and corresponding AIC
    parameters_list:list with (p,q,P,Q) tuples
    d:integration order in ARIMA model
    D:seasonal integration order
    s:length of season
    """
    results = []
    best_aic = float('inf')

    for param in tqdm_notebook(parameters_list):
        # we need try-exccept because on some combinations model fails to converge
        try:
            model = sm.tsa.statespace.SARIMAX(ads.Ads, order=(param[0], d, param[1]),
                                              seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    return result_table


# ------------------------------------------------------------------------------------

result_table = optimizeSARIMA(parameters_list, d, D, s)

# -------------------------------------------------------------------------------------
result_table.head()

# set the parameters that give the lowerst AIC
p, q, P, Q = result_table.parameters[0]
best_model = sm.tsa.statespace.SARIMAX(ads.Ads, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())
best_model.tvalues

# inspect the residuals of the model
tsplot(best_model.resid[24 + 1:], lags=60)


# -------------------------------------------------------------------------------------
def plotSARIMA(series, model, n_steps):
    """
    plot model vs predicted values
    series:dataset with timeseries
    model:fitted SARIMA model
    n_steps:number of steps to predict in the future
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model due the differentiating
    data['arima_model'][:s + d] = np.nan

    # forecasting on n_steps forward
    forecast = model.predict(start=data.shape[0], end=data.shape[0] + n_steps)
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s + d:], data['arima_model'][s + d:])

    plt.figure(figsize=(15, 7))
    plt.title('Mean Absolute Percentage Error: {0:.2f}%'.format(error))
    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label='actual')
    plt.legend()
    plt.grid(True)
    plt.show()


# -------------------------------------------------------------------------------------
plotSARIMA(ads, best_model, 50)

# =====================================================================================
