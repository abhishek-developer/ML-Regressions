import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'C:\Users\Dell pc\OneDrive\Desktop\Projects\LRP\Startups.csv')

#dataset = dataset.apply(pd.to_numeric, errors='coerce')

X = dataset.iloc[:,:-1]

y = dataset.iloc[:,4]

X = pd.get_dummies(X, dtype= int)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)





slope = regressor.coef_

slope

cons = regressor.intercept_

cons

bias = regressor.score(X_train, y_train)

bias

variance = regressor.score(X_test, y_test)
variance

#till here model build

import statsmodels.formula.api as sm


X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) 


import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,4,5]]

#ordinaryleastsquares
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()

regressor_OLS.summary()


import statsmodels.api as sm

X_opt = X[:,[0,1,2,3,5]]

#ordinaryleastsquares
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()

regressor_OLS.summary()

import statsmodels.api as sm

X_opt = X[:,[0,1,2,3]]

#ordinaryleastsquares
regressor_OLS = sm.OLS(endog = y, exog=X_opt).fit()

regressor_OLS.summary()



