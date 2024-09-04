import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Read the data file(csv)
dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset)

#Seprate it to intput(x) and output(y)
x=dataset.iloc[:,1:4].values
y=dataset.iloc[:,-1].values

#change the gender M,F to 0,1
ct=ColumnTransformer([('gender',OneHotEncoder(),[0])],remainder="passthrough")
x=ct.fit_transform(x)

#split the data to train and test 
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

#Sacle the values 
sc=StandardScaler()
x_train =sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#build the model
svc = SVC(kernel='rbf')
svc.fit(x_train,y_train)

#predict 
y_pred=svc.predict(x_test)

#evaluate 
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test, y_pred)
print('the accurancy is',ac*100,'%')