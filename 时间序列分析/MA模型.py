# -*- coding: utf-8 -*-
# 引入相关的统计包
import warnings  # 忽略警告

import numpy as np  # 矢量和矩阵
import pandas as pd  # 表格和数据操作
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
ads = pd.read_csv(r'./ads.csv', index_col=['Time'], parse_dates=['Time'])

warnings.filterwarnings('ignore')


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
    series: dataframe with timeseries
    window:rolling window size
    plot_intervals:show confidence interval
    plot_anomalies:show anomalies
    """
    rolling_mean = series.rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.title('Moving average\n window size={}'.format(window))
    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    # plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, 'r--', label='Upper Bond / Lower Bond')
        plt.plot(lower_bond, 'r--')

    # Having the intervals, find abnormal values
    if plot_anomalies:
        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
        anomalies[series < lower_bond] = series[series < lower_bond]
        anomalies[series > upper_bond] = series[series > upper_bond]
        plt.plot(anomalies, 'ro', markersize=10)

    plt.plot(series[window:], label='Actual values')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

# -------------------------------------------------------------------------------------
# 绘制不同window下的移动平均值和真实值，会发现随着window的增加，移动平均曲线更加平滑
plotMovingAverage(ads, 4)  # smooth by the previous 4 hours
plotMovingAverage(ads, 12)  # smooth by the previous 12 hours
plotMovingAverage(ads, 24)  # smooth by the previous 24 hours, get daily trend
plotMovingAverage(ads, 4, plot_intervals=True)  # 绘制置信区间，查看是否有异常值

# -------------------------------------------------------------------------------------
# 曲线基本正常，故意创造一个含异常值的序列 ads_anomaly
ads_anomaly = ads.copy()
ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20] * 0.2
plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)
