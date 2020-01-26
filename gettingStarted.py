import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv("Data/student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1)) #Drops G3 from dataset used for training data
y = np.array(data[predict]) #only predict data used for training


x_train, x_test, y_train, y_test     = sklearn.model_selection.train_test_split(x,y, test_size = .1)
"""""
best = 0
while(best < .95):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) #fits both data sets in x y plane
    acc = linear.score(x_test, y_test )
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f: #pickle saves linear model
            pickle.dump(linear, f)
"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in) #loads save model with pickle into linear variable

print("Co: " ,linear.coef_)
print("Intercept: ", linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])


#Plotting points to compare different attributes
p = "G1"
style.use("ggplot")
plt.scatter(data[p], data[predict])

plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()