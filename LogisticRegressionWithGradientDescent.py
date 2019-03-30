# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 04:45:33 2019

@author: rajui
"""

# Logistic Regression

# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#This function is used to load CSV file from the 'data' directory 
#in the present working directly 
def loadCSV (fileName):
    scriptDirectory = os.path.dirname(__file__)
    dataDirectoryPath = "."
    dataDirectory = os.path.join(scriptDirectory, dataDirectoryPath)
    dataFilePath = os.path.join(dataDirectory, fileName)
    return pd.read_csv(dataFilePath)

#This funtion is used to preview the data in the given dataset
def previewData (dataSet):
    print(dataSet.head())
    print("\n")

#This function is used to check for missing values in a given dataSet
def checkForMissingValues (dataSet):
    print(dataSet.isnull().sum())
    print("\n")

#This function is used to check the statistics of a given dataSet
def getStatisticsOfData (dataSet):
    print("***** Datatype of each column in the data set: *****")
    dataSet.info()
    print("\n")
    print("***** Columns in the data set: *****")
    print(dataSet.columns.values)
    print("***** Details about the data set: *****")
    print(dataSet.describe())
    print("\n")
    print("***** Checking for any missing values in the data set: *****")
    checkForMissingValues(dataSet)
    print("\n")

#This funtion is used to handle the missing value in the features, in the 
#given examples
def handleMissingValues (feature):
    feature = np.array(feature).reshape(-1, 1)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(feature)
    feature_values = imputer.fit_transform(feature)
    return feature_values
    
#This sigmoid function is responsible for predicting or 
#classifying a given input. It takes input and returns an output 
#of probability, a value between 0 and 1
def sigmoid(X, weight):
    z = np.dot(X, weight)
    return 1 / (1 + np.exp(-z))

#Our objective is to find out the weights (Theta) values, 
#such that the logistic regression function is optimized. 
#Whether the algorithm is performing well or not is defined by a cost function
def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

#Gradient descent is an optimization algorithm that can be used to find the 
#local minimum of a function. So, in our scenario we will use the 
#Gradient descent algorithm to find the minimum of the cost function
def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

#In each iteration of the gradient descent function we will compute 
#the new values for weights, this new values will be used in next iteration
def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

# Importing the dataset
dataset = pd.read_csv('social_network_ads.csv')
#We are using only Age and Estimated value, two parameters in our dataset
X = dataset.iloc[:, [2, 3]].values
#Purshased is the label
y = dataset.iloc[:, 4].values

#Preview the dataSet and look at the statistics of the dataSet
#Check for any missing values 
#so that we will know whether to handle the missing values or not
#print("***** Preview the dataSet and look at the statistics of the dataSet *****")
#previewData(dataset)
#getStatisticsOfData(dataset)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Defining the number of iterations in our Gradient Descent
num_iter = 500

#Defining the learning rate
learning_rate = 0.1

#Defining the intercept, this will be added into the dataset as a column
intercept = np.ones((x_train.shape[0], 1))
x_train = np.concatenate((intercept, x_train), axis=1)
#Defining the initial values for thetas, to be 0
theta = np.zeros(x_train.shape[1])

for i in range(num_iter):
    h = sigmoid(x_train, theta)
    gradient = gradient_descent(x_train, h, y_train)
    theta = update_weight_loss(theta, learning_rate, gradient)
    print("Iteration: {}, Gradient: {}, Thetas: {}".format(i, gradient, theta))

#result will contain the results of classification
result = sigmoid(x_train, theta)

predictedResultDataFrame = pd.DataFrame(np.around(result, decimals=6), columns = ['pred'])
yTrainDataFrame = pd.DataFrame(np.around(y_train, decimals=6), columns = ['class'])
f = pd.concat([predictedResultDataFrame, yTrainDataFrame], axis=1)
f['pred'] = f['pred'].apply(lambda x : 0 if x < 0.5 else 1)
print("\n***** Results from gradient descent *****")
print("Computed coefficients are: ",theta)
print("Accuracy (Loss minimization):", f.loc[f['pred']==f['class']].shape[0] / f.shape[0] * 100)
print("\n")

#*****************************************************************************
# Logistic  Regression using inbuilt sklearn libraries
# Fitting Logistic Regression to the Training set
# Importing the dataset
dataset = pd.read_csv('social_network_ads.csv')
#We are using only Age and Estimated value, two parameters in our dataset
X = dataset.iloc[:, [2, 3]].values
#Purshased is the label
y = dataset.iloc[:, 4].values

#Preview the dataSet and look at the statistics of the dataSet
#Check for any missing values 
#so that we will know whether to handle the missing values or not
#print("***** Preview the dataSet and look at the statistics of the dataSet *****")
#previewData(dataset)
#getStatisticsOfData(dataset)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logisticRegressor = LogisticRegression(random_state = 0)
trainingModelResults = logisticRegressor.fit(x_train, y_train)
print ("***** Results from sklearn.linear_model.LogisticRegression *****")
print ("Coefficients are: ",trainingModelResults.coef_)
print ("Mode accuracy: ",trainingModelResults.score(x_train, y_train)*100)
print ("\n")

# Predicting the Test set results
y_pred = logisticRegressor.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""
"""
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, logisticRegressor.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""