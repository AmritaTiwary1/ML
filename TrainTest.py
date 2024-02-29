#firstOfAll pip install scikit-learn,matplotlib in command prompt if modules are not present
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model  #importing datasets from sklearn so that we can use sets of data to train and test coding
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()  #datasets contains diff. types of data of diabetics keysSet,weather keysSet,face detection keySets,etc

#print(diabetes.keys())  #to print keys of diabetes datasets
# OUTPUT : dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

#print(diabetes.target)   #[151.  75. 141. 206. 135.  97. 138.  63. 110. 310. 101.  69. 179. 185. ........ and many more
#print(diabetes.data.size)  #4420
#rint(diabetes.target.size) #442
#print(diabetes.frame.size())  #AttributeError: 'NoneType' object has no attribute 'size'
#print(diabetes.feature_names)  #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
#print(diabetes.target_filename)  #diabetes_target.csv.gz

'''diabetes_X = diabetes.data[:,np.newaxis,2]  #:,np.newaxis,2 means taking features of 2nd index , np=numpy
#print(diabetes_X)
diabetes_X_train = diabetes_X[0:30]   
#print(diabetes_X_train)
diabetes_X_test = diabetes_X[30:60]
#print(diabetes_X_test)
diabetes_Y_train=diabetes.target[0:30] #target is the result of the data given
diabetes_Y_test=diabetes.target[30:60]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train , diabetes_Y_train)

 diabetes_Y_Predict = model.predict(diabetes_X_test)

print("Mean squared error is : ", mean_squared_error(diabetes_Y_test, diabetes_Y_Predict))

print("weights ",model.coef_)
print("Intercept :",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_Predict)
plt.show() '''
'''OUTPUT : Mean squared error is :  4479.259412797243
weights  [458.15762507]
Intercept : 144.70176785409558'''

diabetes_X = diabetes.data  #diabetes.data means taking all features given in data not only of 2nd index
#print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]
#print(diabetes_X_train)
diabetes_X_test = diabetes_X[-30:]
#print(diabetes_X_test)
diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-30:]
model = linear_model.LinearRegression()
model.fit(diabetes_X_train , diabetes_Y_train)
diabetes_Y_Predict = model.predict(diabetes_X_test) 
#plt cant plot graph bcoz data size currently is 4420 and target size is 442 only 
#plt.scatter(diabetes_X_test,diabetes_Y_test)
#plt.plot(diabetes_X_test,diabetes_Y_Predict)
#plt.show()   #====> output :  raise ValueError("x and y must be the same size")

print("Mean squared error is : ", mean_squared_error(diabetes_Y_test, diabetes_Y_Predict))
print("weights ",model.coef_)
print("Intercept :",model.intercept_)

'''OUTPUT : Mean squared error is :  1826.4841712795044
weights  [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067
  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
Intercept : 153.05824267739402
'''

