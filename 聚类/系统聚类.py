'''以重心法为距离选择方法搭建的系统聚类算法原型'''
# @Feffery
# @说明：目前仅支持维度为2，重心法的情况

import numpy as np
import time

price = [1.1, 1.2, 1.3, 1.4, 10, 11, 20, 21, 33, 34]
increase = [1 for i in range(10)]
data = np.array([price, increase], dtype='float32')


class Myhcluster():

    def __init__(self):
        print('开始进行系统聚类')

    '''系统聚类法的启动函数，有输入变量和距离计算方法两个输入参数'''

    def prepare(self, data, method='zx'):
        if method == 'zx':
            self.zx(data)

    '''重心法进行系统聚类'''

    def zx(self, data):
        token = len(data[0, :])
        flu_data = data.copy()
        classfier = [[] for i in range(len(data[1,]))]
        LSdist = np.array([0 for i in range(token ** 2)], dtype='float32').reshape([len(data[0, :]), token])
        index = 0
        while token > 1:
            '''计算距离矩阵'''
            for i in range(len(data[0, :])):
                for j in range(len(data[0, :])):
                    LSdist[i, j] = round(
                        ((flu_data[0, i] - flu_data[0, j]) ** 2 + (flu_data[1, i] - flu_data[1, j]) ** 2) ** 0.5, 4)

            '''将距离矩阵中的0元素替换为NAN'''
            for i in range(len(data[0, :])):
                for j in range(len(data[0, :])):
                    if LSdist[i, j] == 0:
                        LSdist[i, j] = np.nan

            '''保存该次系统聚类中最短距离对应的两个样本的标号'''
            T = set([np.argwhere(LSdist == np.nanmin(LSdist))[0, 0], np.argwhere(LSdist == np.nanmin(LSdist))[0, 1]])
            TT = [i for i in T]

            '''针对该次聚类情况进行产生新子类亦或是归入旧子类的选择'''
            RQ = TT
            for x in range(len(classfier)):
                if classfier[0] == []:  # 判断是否为n个样本中第一次迭代产生新类
                    classfier[0] = TT
                    index = 0
                    break
                elif classfier[-2] != []:  # 判断是否已在理论最大归类次数前完成所有样品的聚类
                    print('最后一次分类，获得由样本{}组成的新类'.format([__ for __ in range(len(data[1,]))]))
                    return 0
                elif TT[0] in classfier[x] or TT[1] in classfier[x]:
                    if classfier[x + 1] == []:
                        classfier[x + 1] = list(set(classfier[x]).union(set(RQ)))
                        index = x + 1
                        break
                    else:
                        RQ = list(set(classfier[x]).union(set(RQ)))
                        classfier[len(data[1,]) - token] = RQ
                        continue
                elif x == len(data[1,]) - 1:
                    classfier[len(data[0, :]) - token] = TT
                    index = len(data[0, :]) - token
            print('第{}次分类，获得由样本{}组成的新类'.format(str(len(data[0, :]) - token + 1), set(classfier[index])))
            # 求得重心并对原数据进行覆盖
            for k in set(classfier[index]):
                flu_data[0, k] = np.mean([data[0, _] for _ in set(classfier[index])])
                flu_data[1, k] = np.mean([data[1, _] for _ in set(classfier[index])])
            token -= 1


a = time.clock()
dd = Myhcluster()  # 进行算法封装的类的传递
dd.prepare(data)  # 调用类中的系统聚类法（默认重心法）
print('自己编写的系统聚类算法使用了' + str(round(time.clock() - a, 3)) + '秒')
