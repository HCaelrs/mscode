# encoding:utf-8
import numpy as np
from numpy import *

# X=[ [1,2,1,1],
#     [3,3,1,2],
#     [3,5,4,3],
#     [5,4,5,4],
#     [5,6,1,5],
#     [6,5,2,6],
#     [8,7,1,2],
#     [9,8,3,7]]
# X=np.array(X).T#这里注意，[1,2,1,1]在numpy的眼中是一列


np.linalg.eig
X = [[-1, 1, 0],
     [-4, 3, 0],
     [1, 0, 2]]

print("X=", X)
X = matrix(X)

print
"------------------下面计算原始矩阵的特征值和特征向量-----------------------"
eigenvalue, featurevector = np.linalg.eig(X)
print("原始矩阵的特征值")
print("eigenvalue=", eigenvalue)
print("featurevector=", featurevector)