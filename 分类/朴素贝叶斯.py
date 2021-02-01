# 可以通过离散的数据直接尽心贝叶斯分类
#　如果是连续的数据，则需要通过其他的数据估计模型进行查找　
#　数据离散化　正泰分布的贝叶斯分类器
# GaussianNB 实现了运用于分类的高斯朴素贝叶斯算法
from sklearn import datasets
"""
Attributes:	
`class_prior_` : array, shape = [n_classes]

probability of each class.

`theta_` : array, shape = [n_classes, n_features]

mean of each feature per class

`sigma_` : array, shape = [n_classes, n_features]

variance of each feature per class
"""