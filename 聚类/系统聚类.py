'''�����ķ�Ϊ����ѡ�񷽷����ϵͳ�����㷨ԭ��'''
# @Feffery
# @˵����Ŀǰ��֧��ά��Ϊ2�����ķ������

import numpy as np
import time

price = [1.1, 1.2, 1.3, 1.4, 10, 11, 20, 21, 33, 34]
increase = [1 for i in range(10)]
data = np.array([price, increase], dtype='float32')


class Myhcluster():

    def __init__(self):
        print('��ʼ����ϵͳ����')

    '''ϵͳ���෨����������������������;�����㷽�������������'''

    def prepare(self, data, method='zx'):
        if method == 'zx':
            self.zx(data)

    '''���ķ�����ϵͳ����'''

    def zx(self, data):
        token = len(data[0, :])
        flu_data = data.copy()
        classfier = [[] for i in range(len(data[1,]))]
        LSdist = np.array([0 for i in range(token ** 2)], dtype='float32').reshape([len(data[0, :]), token])
        index = 0
        while token > 1:
            '''����������'''
            for i in range(len(data[0, :])):
                for j in range(len(data[0, :])):
                    LSdist[i, j] = round(
                        ((flu_data[0, i] - flu_data[0, j]) ** 2 + (flu_data[1, i] - flu_data[1, j]) ** 2) ** 0.5, 4)

            '''����������е�0Ԫ���滻ΪNAN'''
            for i in range(len(data[0, :])):
                for j in range(len(data[0, :])):
                    if LSdist[i, j] == 0:
                        LSdist[i, j] = np.nan

            '''����ô�ϵͳ��������̾����Ӧ�����������ı��'''
            T = set([np.argwhere(LSdist == np.nanmin(LSdist))[0, 0], np.argwhere(LSdist == np.nanmin(LSdist))[0, 1]])
            TT = [i for i in T]

            '''��Ըôξ���������в�������������ǹ���������ѡ��'''
            RQ = TT
            for x in range(len(classfier)):
                if classfier[0] == []:  # �ж��Ƿ�Ϊn�������е�һ�ε�����������
                    classfier[0] = TT
                    index = 0
                    break
                elif classfier[-2] != []:  # �ж��Ƿ������������������ǰ���������Ʒ�ľ���
                    print('���һ�η��࣬���������{}��ɵ�����'.format([__ for __ in range(len(data[1,]))]))
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
            print('��{}�η��࣬���������{}��ɵ�����'.format(str(len(data[0, :]) - token + 1), set(classfier[index])))
            # ������Ĳ���ԭ���ݽ��и���
            for k in set(classfier[index]):
                flu_data[0, k] = np.mean([data[0, _] for _ in set(classfier[index])])
                flu_data[1, k] = np.mean([data[1, _] for _ in set(classfier[index])])
            token -= 1


a = time.clock()
dd = Myhcluster()  # �����㷨��װ����Ĵ���
dd.prepare(data)  # �������е�ϵͳ���෨��Ĭ�����ķ���
print('�Լ���д��ϵͳ�����㷨ʹ����' + str(round(time.clock() - a, 3)) + '��')
