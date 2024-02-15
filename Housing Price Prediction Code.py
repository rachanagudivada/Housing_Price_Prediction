#!/usr/bin/env python
# coding: utf-8

# import Packages 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


from pandas import read_csv
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'price']
dataset = read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)
print(dataset.head())

dataset.info()

dataset.describe()

dataset.isnull().sum()

# split the Independent and dependent features

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1 ]
X
y

# train_test_split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)
X_train
X_test


# Linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
y_pred_linear

print('Linear Regression:')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

print('Mean Squared error : ', np.sqrt(mean_squared_error(y_test, y_pred_lin)))
print('Mean Absolute error: ', mean_absolute_error(y_test, y_pred_lin))


from sklearn.metrics import r2_score
linear_score = r2_score(y_test, y_pred_linear)
print(linear_score)

# plot a Scatter plot for prediction and y_test)
plt.scatter(y_test, y_pred_linear)
plt.xlabel('True Value ')
plt.ylabel('Predict value')
plt.show()

# residuals
residuals = y_test - y_pred_linear 
residuals

# plot the residuals
sns.displot(residuals, kind='kde')
plt.show()

# plot a Scatter plot for prediction and Residuals
plt.scatter(y_pred_linear, residuals)
plt.show()

# Random forest regressor
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
y_pred_rf
print('RandomForest Regressor:')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

print('Mean Squared error : ', np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print('Mean Absolute error: ', mean_absolute_error(y_test, y_pred_rf))

from sklearn.metrics import r2_score
rf_score = r2_score(y_test, y_pred_rf)
print(rf_score)

# plot a Scatter plot for prediction and y_test)
plt.scatter(y_test, y_pred_rf)
plt.xlabel('True Value ')
plt.ylabel('Predict value')
plt.show()

# residuals
residuals = y_test - y_pred_rf 
residuals
# plot the residuals
sns.displot(residuals, kind='kde')
plt.show()
# plot a Scatter plot for prediction and Residuals
plt.scatter(y_pred_rf, residuals)
plt.show()
# SVM regressor
from sklearn.svm import SVR
svm_reg = SVR(kernel='rbf') 
svm_reg.fit(X_train, y_train)
y_pred_svm = svm_reg.predict(X_test)
y_pred_svm
print('SVM Regressor:')
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

print('Mean Squared error : ', np.sqrt(mean_squared_error(y_test, y_pred_svm)))
print('Mean Absolute error: ', mean_absolute_error(y_test, y_pred_svm))

from sklearn.metrics import r2_score
svm_score = r2_score(y_test, y_pred_svm)
print(svm_score)

# plot a Scatter plot for prediction and y_test)
plt.scatter(y_test, y_pred_svm)
plt.xlabel('True Value ')
plt.ylabel('Predict value')
plt.show()

# residuals
residuals = y_test - y_pred_svm 
residuals

# plot the residuals
sns.displot(residuals, kind='kde')
plt.show()


# plot a Scatter plot for prediction and Residuals
plt.scatter(y_pred_svm,residuals)
plt.show()

# Convert R-squared to percentage
acc_lin = linear_score * 100
acc_rf = rf_score * 100
acc_svm = svm_score* 100

print('Linear Regression Accuracy: {:.2f}%'.format(acc_lin))
print('Random Forest Regression Accuracy: {:.2f}%'.format(acc_rf))   
print('SVM Regression Accuracy: {:.2f}%'.format(acc_svm))




