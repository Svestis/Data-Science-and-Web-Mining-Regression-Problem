# Data-Science-and-Web-Mining-Regression-Problem

# The problem
As part of the course "Data science & Web Mining", you will be working on a regression problem. Specifically, you are given a dataset consisting of a few thousand records containing the hourly count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information.
Your goal is to predict how many bikes will be rented each hour of a day, based on data including weather, time, temperature, whether or not its a workday, and much more.
The dataset was taken from UCI Machine Learning Repository.

# Solution

The best learning model that we achieved for this project with this dataset is the stacking model by considering all features. Stacking was performed with multiple models and the most successful was the one with two different GradientBoosting models, one Random Forest model, one SVR model and one ExtraTreesRegressor.

This model fits well for both the training and test sets and is not over fitting as per below.
