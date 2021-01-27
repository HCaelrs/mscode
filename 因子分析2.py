# encoding:utf-8
import pandas as pd
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity


def main():
    # excelFile = 'data.xlsx'
    # df = pd.DataFrame(pd.read_excel(excelFile))
    # df2 = df.copy()
    # print("\n原始数据:\n", df2)
    # del df2['ID']
    df = pd.read_csv("./TestData/datasets/iris_train.csv")
    # print(df)
    df2 = df.copy()
    del df2['Species']

    print("\n原始数据:\n", df2)

    # 皮尔森相关系数
    df2_corr = df2.corr()
    print("\n相关系数:\n", df2_corr)

    kmo = calculate_kmo(df2)  # kmo值要大于0.7
    bartlett = calculate_bartlett_sphericity(df2)  # bartlett球形度检验p值要小于0.05
    print("\n因子分析适用性检验:")
    print('kmo:{},bartlett:{}'.format(kmo[1], bartlett[1]))

    fa = FactorAnalyzer(rotation=None, n_factors=15, method='principal')
    fa.fit(df2)
    fa_15_sd = fa.get_factor_variance()
    fa_15_df = pd.DataFrame(
        {'特征值': fa_15_sd[0], '方差贡献率': fa_15_sd[1], '方差累计贡献率': fa_15_sd[2]})

    # 各个因子的特征值以及方差贡献率
    print("\n", fa_15_df)

    # 公因子数设为5个，重新拟合
    fa_5 = FactorAnalyzer(rotation=None, n_factors=5, method='principal')
    fa_5.fit(df2)

    # 查看公因子提取度
    print("\n公因子提取度:\n", fa_5.get_communalities())

    # 查看因子载荷
    print("\n因子载荷矩阵:\n", fa_5.loadings_)

    # 使用最大方差法旋转因子载荷矩阵
    fa_5_rotate = FactorAnalyzer()
    fa_5_rotate.fit(df2)

    # # 查看旋转后的因子载荷
    # print("\n旋转后的因子载荷矩阵:\n", fa_5_rotate.loadings_)
    #
    # # 因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
    # X1 = np.mat(df2_corr)
    # X1 = nlg.inv(X1)
    #
    # # B=(R-1)*A  15*5
    # factor_score = np.dot(X1, fa_5_rotate.loadings_)
    # factor_score = pd.DataFrame(factor_score)
    # factor_score.columns = ['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    # factor_score.index = df2_corr.columns
    # print("\n因子得分：\n", factor_score)
    #
    # # F=XB  27*15 15*5=  27 5
    # fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
    # print("\n应试者的五个因子得分：\n", pd.DataFrame(fa_t_score))
    #
    # # 综合得分(加权计算）
    # wei = [[0.378637], [0.224112], [0.096413], [0.082957], [0.059127]]
    # fa_t_score = np.dot(fa_t_score, wei) / 0.841246
    # fa_t_score = pd.DataFrame(fa_t_score)
    # fa_t_score.columns = ['综合得分']
    # fa_t_score.insert(0, 'ID', range(1, 28))
    # print("\n综合得分：\n", fa_t_score)
    # print("\n综合得分：\n", fa_t_score.sort_values(by='综合得分', ascending=False).head())
    #
    # ax1 = plt.subplot(111)
    # X = fa_t_score['ID']
    # Y = fa_t_score['综合得分']
    # plt.bar(X, Y, color="red")
    # plt.title('result00')
    # ax1.set_xticks(range(len(fa_t_score)))
    # ax1.set_xticklabels(fa_t_score.index)
    # plt.show()
    #
    # fa_t_score1 = pd.DataFrame()
    # fa_t_score1 = fa_t_score.sort_values(by='综合得分', ascending=False).head()
    # X1 = fa_t_score1['ID']
    # Y1 = fa_t_score1['综合得分']
    # plt.bar(X1, Y1, color='red')
    # plt.title('result01')
    # plt.show()


if __name__ == '__main__':
    main()
