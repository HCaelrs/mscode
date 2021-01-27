# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression
import numpy as np


def colicSklearn():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []

    # 读取文件的所有行 返回的是一个列表
    allLines = frTrain.readlines()
    numLines = len(allLines)

    allList = np.arange(299)

    trainingList = np.random.choice(numLines, int(0.999*numLines),replace=False)
    testList = set(allList)-set(trainingList)

    print(testList)
    print(set(trainingList))
    #
    # for line in frTrain.readlines():
    #     currLine = line.strip().split('\t')

    for i in trainingList:
        currLine = allLines[i].strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for i in testList:
        currLine = allLines[i].strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    classifier = LogisticRegression(solver='liblinear', max_iter=100).fit(trainingSet, trainingLabels)
    test_accuracy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accuracy)


if __name__ == '__main__':
    colicSklearn()
