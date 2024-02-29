import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model  #importing datasets from sklearn so that we can use sets of data to train and test coding
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()  #datasets contains diff. types of data of diabetics keysSet,weather keysSet,face detection keySets,etc

diabetes_X = np.array([[1],[2],[3]])

diabetes_X_train = diabetes_X
diabetes_Y_train = np.array([3,2,4])

diabetes_X_test = diabetes_X
diabetes_Y_test = np.array([3,2,4])

model = linear_model.LinearRegression()
model.fit(diabetes_X_train , diabetes_Y_train)

diabetes_Y_Predict = model.predict(diabetes_X_test)

print("Mean squared error is : ", mean_squared_error(diabetes_Y_test, diabetes_Y_Predict))

print("weights ",model.coef_)
print("Intercept :",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_Predict)
plt.show()

'''OUTPUT : Mean squared error is :  0.5000000000000001
weights  [0.5]
Intercept : 2.0
y=mx+c ,Exact same as the value of M (weights) and C (intercepts) m=1/2 & c=2 '''
