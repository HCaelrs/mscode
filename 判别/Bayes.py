import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train_data = pd.read_excel('HCV_train.xls', header=None)
train_feature = train_data[train_data.columns[:train_data.shape[1] - 1]].values
train_label = list(train_data[train_data.columns[train_data.shape[1] - 1]])
clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=7,
                                  max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=2,
                                  min_weight_fraction_leaf=0.0, presort=False, random_state=1, splitter='random')
# clf = GaussianNB()
# clf=KNeighborsClassifier(2)
# clf=GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)

clf = clf.fit(train_feature, train_label)

test_data = pd.read_excel('HCV_test.xls', header=None)
test_feature = test_data[test_data.columns[:test_data.shape[1] - 1]].values
test_label = list(test_data[test_data.columns[test_data.shape[1] - 1]])
test_pred = clf.predict(test_feature)
print(np.mean(test_pred == test_label))
print(confusion_matrix(test_label, test_pred))
print(classification_report(test_label, test_pred))
