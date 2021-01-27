# encoding:utf-8
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def pca(X, d):
    # Centralization
    # means = np.mean(X, 0)
    # X = X - means
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    # 上面两部是在进行标准化

    # Covariance Matrix

    covM = np.dot(X.T, X)
    # 将X和X的转置相乘得到covM

    eigval, eigvec = np.linalg.eig(covM)
    # 获得特征值和特征向量

    indexes = np.argsort(eigval)[-d:]
    """
    argsort 的作用是将所有的进行排序，但是其排序完成后显示的不是数字的大小而是其位置
    
    """
    # 对其按列进行排序

    W = eigvec[:, indexes]
    print(W)
    return np.dot(X, W)


iris = datasets.load_iris()
X_pca, w = pca(iris.data, 2)
print(w)
# 只要传入X就可以 不需要Y的传入 传入的仅仅是数据，没有其中的结果
kmeans = KMeans(n_clusters=3).fit(X_pca)
plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap=plt.cm.Set1)
plt.show()