import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#Reading the data file
data = pd.read_csv("C:/Users/Google/Desktop/python/AI project/assets/student-mat.csv",sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.3)
#Training the model and saving the best one 
'''
best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x ,y, test_size=0.3)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentsmodel.pickle", "wb") as f:
            pickle.dump(linear,f)
print(f"___________{best}_____________")
'''
pickle_in = open("studentsmodel.pickle", "rb")
linear = pickle.load(pickle_in)

#testing the model by predicting the x_test
predictions = linear.predict(x_test)

'''for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])'''
#visuallizing the data and its correlation 
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()