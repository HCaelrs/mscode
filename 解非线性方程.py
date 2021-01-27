# coding:utf-8
from scipy import optimize
import math

"""
计算非线性方程组：
    5x1+3 = 0
    4x0^2-2sin(x1x2)=0
    x1x2-1.5=0
"""


## 误差函数
def fun(x):
    x0, x1, x2 = x.tolist()
    return [5 * pow(x1, x2) - 3, 4 * pow(x0, 2) - 2 * math.sin(x1 * x2), x1 * x2 - 1.5]


result = optimize.fsolve(fun, [1, 1, 1])
print(result)
# result
# [-0.70622057    -0.6    -2.5]
