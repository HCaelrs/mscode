# encoding:utf-8
from statsmodels.tsa import stattools
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("ads.csv")
inpt = df.Ads
print('自相关系数: \n', stattools.acf(inpt))
print('偏自相关系数: \n', stattools.pacf(inpt))
# 可以看到，这里自相关系数和偏自相关系数的计算都返回了长度为10的数组。它们分别代表了当l ll的取值为从0到10时的自相关性统计


# 显示原数据
plt.plot(df.Time, df.Ads)
plt.title('原数据', fontsize=22)

plt.show()

# 显示acf
plt.stem(stattools.acf(inpt))
plt.show()

# 显示pacf
plt.stem(stattools.pacf(inpt))
plt.show()

"""
https://blog.csdn.net/weixin_42382211/article/details/81332431
ADF Test in Python
在python中对时间序列的建模通常使用statsmodel库，该库在我心中的科学计算库排名中长期处于垫底状态，因为早期文档实在匮乏，不过近来似有好转倾向。
在statsmodels.tsa.stattools.adfuller中可进行adf校验，一般传入一个1d 的 array like的data就行，包括list， numpy array 和 pandas series都可以作为输入，其他参数可以保留默认.
其返回值是一个tuple，格式如下：
print sm.tsa.stattools.adfuller(dta)
"""

print(stattools.adfuller(inpt))


"""
如何确定该序列能否平稳呢？主要看：

1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，
ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设，本数据中，adf结果为-9， 小于三个level的统计值。
"""