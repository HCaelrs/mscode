# encoding:utf-8
import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris
import numpy as np


data = load_iris()
y = data.target
x = data.data
x -= np.mean(x, axis=0)
x /= np.std(x, axis=0)
pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2

Z = pca.fit_transform(x)  # 对样本进行降维

# w = Z/x
# print(w)
# z=wx w = z(x^-1)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(Z)):
    if y[i] == 0:
        red_x.append(Z[i][0])
        red_y.append(Z[i][1])
    elif y[i] == 1:
        blue_x.append(Z[i][0])
        blue_y.append(Z[i][1])
    else:
        green_x.append(Z[i][0])
        green_y.append(Z[i][1])
# 可视化
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

