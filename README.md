# Diabetes
# Exploration and visualization of diabetes risk factors

#Import packages 
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Import data and define
d1 = pd.read_excel('/Users/christinacathey/Downloads/Final Project Health Data Set/diabetes_012_health_indicators_BRFSS2015.xlsx')

d1.shape

d1.head()

#Transform data
for i in d1.columns.tolist():
    d1[i] = d1[i].astype('int')

d1.head()

d1.info()

d1.describe()

#Explore correlation between data using a heatmap
plt.figure(figsize = (12,8))
sns.heatmap(d1.corr(), vmax = 1, square = True)
plt.title("Diabetes Correlation Heatmap")
plt.show()

#Split data to set it up for training and testing
X = d1.drop('Diabetes_012', axis = 1)
y = d1['Diabetes_012']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Build a Random Forest Model
d1.rf = RandomForestClassifier(random_state = 1, max_features = 'sqrt', n_jobs = 1, verbose = 1)
%time d1.rf.fit(X_train, y_train)
d1.rf.score(X_test, y_test)

#Model Predictions
y_prediction = d1.rf.predict(X_test)
print(y_prediction)

#Check Mean Squared and Root Mean Squared Error
d1_mse = metrics.mean_squared_error(y_test, y_prediction)
print('Mean Squared Error : '+ str(d1_mse))
d1_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_prediction))
print('Root Mean Squared Error : '+ str(d1_rmse))

#Create a confusion matrix
d1_matrix = metrics.confusion_matrix(y_test, y_prediction)
print(d1_matrix)


plt.figure(figsize = (8,6))
sns.heatmap(d1_matrix, annot = True, fmt = ".0f", cmap = 'viridis')
plt.title("Diabetes Confusion Matrix")
plt.xlabel("Predictions")
plt.ylabel("Actual")
plt.show()

#Create report
d1_report = metrics.classification_report(y_test, y_prediction)
print(d1_report)

#Data Visualization 
plt.figure(figsize = (10,5))
sns.countplot(d1['Diabetes_012'])
plt.title("No Diabetes, Pre- Diabetes, and Diabetes")
plt.show()

#Group diabetes status and BMI
diabetes_bmi = d1.groupby(['Diabetes_012', 'BMI']).size().reset_index(name = 'Count')
print(diabetes_bmi)

#Visualize diabetes status and BMI
plt.figure(figsize = (10,5))
sns.barplot(x = 'Diabetes_012', y = 'Count', hue = 'BMI', data = diabetes_bmi, palette = 'Set1')
plt.title("Dibaetes Status and BMI")
plt.show()

#Group diabetes status and Smoker
diabetes_smoker = d1.groupby(['Diabetes_012', 'Smoker']).size().reset_index(name = 'Count')
print(diabetes_smoker)

#Visualize diabetes status and Smoker
plt.figure(figsize = (10,5))
sns.barplot(x = 'Diabetes_012', y = 'Count', hue = 'Smoker', data = diabetes_smoker, palette = 'Set1')
plt.title("Dibaetes Status and Smoking Status")
plt.show()

#Group diabetes status and HighBP
diabetes_bp = d1.groupby(['Diabetes_012', 'HighBP']).size().reset_index(name = 'Count')
print(diabetes_bp)

#Visualize diabetes status and HighBP
plt.figure(figsize = (10,5))
sns.barplot(x = 'Diabetes_012', y = 'Count', hue = 'HighBP', data = diabetes_bp, palette = 'Set1')
plt.title("Dibaetes Status and High Blood Pressure")
plt.show()

#Group diabetes status and Healthcare
diabetes_healthcare = d1.groupby(['Diabetes_012', 'AnyHealthcare']).size().reset_index(name = 'Count')
print(diabetes_healthcare)

#Visualize diabetes status and Healthcare
plt.figure(figsize = (10,5))
sns.barplot(x = 'Diabetes_012', y = 'Count', hue = 'AnyHealthcare', data = diabetes_healthcare, palette = 'Set1')
plt.title("Diabetes Status and Any Healthcare")
plt.show()

#Group diabetes status and Age
diabetes_age = d1.groupby(['Diabetes_012', 'Age']).size().reset_index(name = 'Count')
print(diabetes_age)

#Visualize diabetes status and Healthcare
plt.figure(figsize = (10,8))
sns.barplot(x = 'Diabetes_012', y = 'Age', hue = 'Age', data = diabetes_age, palette = 'Set2')
plt.title("Diabetes Status and Age")
plt.show()

#Group diabetes status and Income
diabetes_income = d1.groupby(['Diabetes_012', 'Income']).size().reset_index(name = 'Count')
print(diabetes_income)

#Visualize diabetes status and Income
plt.figure(figsize = (10,8))
sns.barplot(x = 'Diabetes_012', y = 'Income', hue = 'Income', data = diabetes_income, palette = 'Set3')
plt.title("Diabetes Status and Income")
plt.show()

#Group diabetes status and CholCheck
diabetes_chol = d1.groupby(['Diabetes_012', 'CholCheck']).size().reset_index(name = 'Count')
print(diabetes_chol)

#Visualize diabetes status and CholCheck
plt.figure(figsize = (10,8))
sns.barplot(x = 'Diabetes_012', y = 'CholCheck', hue = 'CholCheck', data = diabetes_chol, palette = 'Set3')
plt.title("Diabetes Status and Cholesterol")
plt.show()

