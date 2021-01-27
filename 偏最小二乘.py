# encoding: utf-8

import numpy as np

#
# # 用csv进行对于数据的读取
# myFile = open('data.csv', 'r+')
# lines = csv.reader(myFile)
# for line in lines:
#     print(line)
#
data = np.loadtxt("data.csv", dtype=float, delimiter=',')
print(data)

y = data[..., 1]
X = data[..., 2:]

y = y.reshape(1, -1).T


# 标准化
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

y -= np.mean(y, axis=0)
y /= np.std(y, axis=0)

print(X)

print("列：\n", y)

y0 = y
X0 = X
yy0 = 0
XX0 = 0
n = 14


for i in range(1, n):
    print(i)
    ta = np.dot(np.dot(X0, X0.T), y0)
    ta = ta.reshape(1, -1).T
    yya = np.dot(np.dot(ta, ta.T) / np.dot(ta.T, ta), y0) + yy0
    # print(y,'and\n',yya,'and\n')
    # print(y-yya)
    # ans = y - yya

    out = np.dot((y-yya).T, (y-yya))
    print(float(out))
    ya = y0 - np.dot(np.dot(ta, ta.T) / np.dot(ta.T, ta), y0)
    XXa = np.dot(np.dot(ta, ta.T) / np.dot(ta.T, ta), X0)
    Xa = X0-XXa

    X0 = Xa
    y0 = ya
    XX0 = XXa
    yy0 = yya

print(y0)
print(X0)
