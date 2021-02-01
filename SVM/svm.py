from sklearn import svm
model = svm.SVC(kernel='linear', C=1, gamma=1)
model.get_params()
