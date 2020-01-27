"""
SVM creates a plane of n-1 dimension in order to separate classes
If the the classes overlap in the current dimension,
then the number of dimensions must be increased
"""

import sklearn
from sklearn import datasets
from sklearn import svm #classifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

classes = ["malignant", "Benign"]

#clf = svm.SVC(kernel="linear", C=2) #C is the soft margin: number of points that can cross the midplane
clf = KNeighborsClassifier(n_neighbors=9) #KNeighbors doesn't work as well with lots of classifiers
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print acc
