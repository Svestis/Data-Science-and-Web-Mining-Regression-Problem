'''
Purpose:

We are given a dataset which contains the hourly count of bike rentals over a period of 12,165 hours
distributed in 731 days and we need to predict how many bikes will be rented each hour of the day,
using various prediction models and techniques.

Approach:
The best learning model that we achieved for this project with this dataset is the stacking model
 considering all features. Stacking was performed Stacking on the other hand generally produced
 better results. We tested stacking with multiple regressors and the most successful was the
 one with two different GradientBoosting models, one Random Forest model, one SVR model
 , one ExtraTreesRegressor.

This model fits well for both the training and test sets and is not over fitting
'''
# Pylint score is 10.00/10

# Importing Needed Libraries

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR

# Reading dataset

TRAIN_DS = pd.read_csv('train.csv')  # Training dataset

# Preparing data
# Changing column names for better reading

TRAIN_DS.rename(columns={'weathersit':'weather',
                         'mnth':'month',
                         'hr':'hour',
                         'yr':'year',
                         'hum': 'humidity',
                         'cnt':'count',
                         'temp':'temperature'}, inplace=True)

# Some data types need to be changed from numerical to categorical
# in order for the model to interpret better these variables.

TRAIN_DS['season'] = TRAIN_DS.season.astype('category')
TRAIN_DS['year'] = TRAIN_DS.year.astype('category')
TRAIN_DS['month'] = TRAIN_DS.month.astype('category')
TRAIN_DS['hour'] = TRAIN_DS.hour.astype('category')
TRAIN_DS['holiday'] = TRAIN_DS.holiday.astype('category')
TRAIN_DS['weekday'] = TRAIN_DS.weekday.astype('category')
TRAIN_DS['workingday'] = TRAIN_DS.workingday.astype('category')
TRAIN_DS['weather'] = TRAIN_DS.weather.astype('category')
TRAIN_DS['temperature'] = TRAIN_DS.temperature.astype('category')

# Dropping not needed categories

TRAIN_DS = TRAIN_DS.drop(['atemp', 'casual', 'registered', 'windspeed'], axis=1)

# Submitting all features for predictions

X = TRAIN_DS[["season", "holiday", "workingday", "weather", "weekday",
              "month", "year", "hour", 'humidity', 'temperature']]

Y = TRAIN_DS['count']

# Creating a list for the models that we will use as estimators - best based on GridSearchCV

ESTIMATORS = [('randf', RandomForestRegressor(max_depth=50, n_estimators=1500)),
              ('gradb', GradientBoostingRegressor(max_depth=5, n_estimators=400)),
              ('gradb2', GradientBoostingRegressor(n_estimators=4000)),
              ('svr', SVR('rbf', gamma='auto')),
              ('ext', ExtraTreesRegressor(n_estimators=4000))]

# Creating the model from the estimators

STACKING = StackingRegressor(ESTIMATORS)

# Fitting the model

STACKING.fit(X, y=np.log1p(Y))

# Creating submission file

# Reading file

DF_TEST = pd.read_csv('test.csv')

# Preparing data
# Changing column names for better reading

DF_TEST.rename(columns={'weathersit':'weather',
                        'mnth':'month',
                        'hr':'hour',
                        'yr':'year',
                        'hum': 'humidity',
                        'cnt':'count',
                        'temp':'temperature'}, inplace=True)

# Some data types need to be changed from numerical to categorical
#in order for the model to interpret better these variables.

DF_TEST['season'] = DF_TEST.season.astype('category')
DF_TEST['year'] = DF_TEST.year.astype('category')
DF_TEST['month'] = DF_TEST.month.astype('category')
DF_TEST['hour'] = DF_TEST.hour.astype('category')
DF_TEST['holiday'] = DF_TEST.holiday.astype('category')
DF_TEST['weekday'] = DF_TEST.weekday.astype('category')
DF_TEST['workingday'] = DF_TEST.workingday.astype('category')
DF_TEST['weather'] = DF_TEST.weather.astype('category')
DF_TEST['temperature'] = DF_TEST.temperature.astype('category')

# Dropping not needed categories

DF_TEST = DF_TEST.drop(['atemp', 'windspeed'], axis=1)
DF_TEST = DF_TEST[["season", "holiday", "workingday", "weather", "weekday",
                   "month", "year", "hour", 'humidity', 'temperature']]

# Making predictions

Y_PRED = STACKING.predict(DF_TEST)
PREDICTIONS = np.exp(Y_PRED)

# Removing 0's

for i, y  in enumerate(PREDICTIONS):
    if Y_PRED[i] < 0:
        Y_PRED[i] = 0

# Creating submission file

SUBMISSION = pd.DataFrame()
SUBMISSION['Id'] = range(PREDICTIONS.shape[0])
SUBMISSION['Predicted'] = PREDICTIONS
SUBMISSION.to_csv("submission.csv", index=False)

# Creating distribution graph

Y = TRAIN_DS['count']
GRAPH, (TRN, TST) = plt.subplots(ncols=2)
GRAPH.set_size_inches(20, 5)
sn.distplot(Y, ax=TRN, bins=100)
sn.distplot(PREDICTIONS, ax=TST, bins=100)
TRN.set(title="Training Set Distbution")
TST.set(title="Test Set Distribution")
