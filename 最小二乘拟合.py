# coding:utf-8

###最小二乘法试验###
import numpy as np
from scipy.optimize import leastsq

###采样点(Xi,Yi)###
Xi = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
Yi = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])


###需要拟合的函数func及误差error###
def func(p, x):
    k0, b0 = p
    return k0 * x + b0


def error(p, x, y):
    return func(p, x) - y
    # x、y都是列表，故返回值也是个列表


# TEST
p0 = [100, 2]
# print( error(p0,Xi,Yi) )

###主函数从此开始###
Para = leastsq(error, [100, 2], args=(Xi, Yi))  # 把error函数中除了p以外的参数打包到args中
k, b = Para[0]
print(Para)
print("k=", k, '\n', "b=", b, sep='')

###绘图，看拟合效果###
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(Xi, Yi, color="red", label="Sample Point", linewidth=3)  # 画样本点
x = np.linspace(0, 10, 1000)
y = k * x + b
plt.plot(x, y, color="orange", label="Fitting Line", linewidth=2)  # 画拟合直线
plt.legend()
plt.show()
