# PROBLEM STATEMENT

### The objective of the project is to predict the quality of the wine
### The following parameters are used to test the wine quality 
### 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol',
### Output : Rate the wine quality on a scale 1-10 or Bad - Good

# Importing libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
import random
import scipy

#Supress the warnings
import warnings
warnings.filterwarnings('ignore')

import pickle

from collections import Mapping

#  Load Dataset

wine_quality = pd.read_csv('QualityPrediction.csv')

#   Pre Processing

wine_quality.head()

wine_quality.shape

wine_quality.describe()

## Nature of variables/ features
# Predictor variables --- Numerical data
# Response variables  --- Categorical ( Ordinal) data

##  List of Features / Columns

features = wine_quality.columns
features

## Unique values in the target feature 'quality'

uni_val = round(((wine_quality['quality'].value_counts()/len(wine_quality['quality']))*100),2)
print(f'Wine quality scales in %:\n\n{uni_val}')

## The target variable 'quality' has ordinal values ranging from 3 to 8. 
## It's also observed that ordinal values 1,2, 9 and 10 are not present in the dataset and the ML model will not be exposed to these range 
## There are two options to train the model
# Multi class classification ( with quality ranging from 3 to 8 only)  --- Decision Tree/Random forest / KNN algorithm
# Binary class calssification with 3,4 and 5 mapped as bad(0) and 6,7 & 8 mapped as good(1)  ---- Logistic Regression/Decision Tree/Random forest / KNN algorithm

##  Missing values 

wine_quality.isnull().sum()      

## There are no missing values in the given dataset

# 2. Checking for outliers

##  Box plot

def box_plot(df,feature):
    df.boxplot(column=[feature],vert=0)
    plt.show()

for i in range(0,wine_quality.shape[1]):
    box_plot(wine_quality,features[i])

## Outlier removal using IQR method

Q1 = wine_quality.quantile(0.25)
Q3 = wine_quality.quantile(0.75)
IQR = Q3 - Q1
pos_outlier = Q3 + 1.5 * IQR
neg_outlier = Q1 - 1.5 * IQR

wine_quality = wine_quality[~((wine_quality < neg_outlier) | (wine_quality > pos_outlier)).any(axis=1)]

wine_quality.shape

### Outliers are removed from dataframe and a clean dataframe is created to train the model 

## Encoding target variable into 2 class variables
#### For Logistic Regression the target variables need to be binary . Here the target variable 'quality' ranges from 3 to 8.
#### Let's transform the categories into two classes 0 and 1
### combining 3,4 & 5 as '0' and 6,7 & 8 as '1' will result in a balanced dataset for efficient training
#### 3,4,5 ----- > 0  
#### 6,7,8 ------> 1

round((wine_quality['quality'].value_counts()/len(wine_quality['quality'])*100),2)

wine_quality['quality'] = wine_quality['quality'].map({3:0,4:0,5:0,6:1,7:1,8:1})
wine_quality.head()

wine_quality['quality'].value_counts()

### Mapping 3,4 & 5 into '0' and 6,7 & 8 into '1' gives balanced dataset for training

# EDA (Exploratory Data Analysis)

sns.countplot(data=wine_quality,x='quality')
Bad,Good=wine_quality['quality'].value_counts()
print("Bad Wine :",Bad)
print("Good Wine:",Good)
plt.show()

### The dataset is balanced

# Feature Scaling
## Robust Scaling 
### Here most of the independent variables (features) are skewed. So We are using Robust Scaling

# Defining Independent and Dependent variables

x = wine_quality.iloc[:,wine_quality.columns!='quality']
y = wine_quality.iloc[:,wine_quality.columns=='quality']

from sklearn.preprocessing import RobustScaler
# Scaling predictor variables and target varibles using Robust Scaler
robust_scaler = RobustScaler()
x_scaled = robust_scaler.fit_transform(x)

# Checking multicolinearity among features

## 1.  VIF (Variance Inflation Factor)

from statsmodels.stats.outliers_influence import variance_inflation_factor
variable = x_scaled

vif = pd.DataFrame()
vif['Features'] = x.columns
vif['Variance Inflation Factor'] = [variance_inflation_factor(variable, i) for i in range(variable.shape[1])]


vif
# If VIF> 5 , the feature has multicolinearity
# The features 'fixed acidity' and 'density' have multicollinearity 

## 2. Correlation Matrix / Heatmap

sns.heatmap(wine_quality.corr(),annot=True,cmap='coolwarm')

wine_quality.corr()

#'fixed acidity' and 'density' have correlation value 0.61

### Conclusion: 
# It is evident from VIF factor and Correltion matrix that the features 'fixed acidity' and 'density' have multicollinearity with each other.

## Spliting taining and testing data set


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75,random_state=100)

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

# Random Forest


# Fitting the model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)


# Saving the nodel 
pickle.dump(rf,open('model.pkl','wb'))

# Load the model to view results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[11.2,0.28,0.56,1.9,0.075,17.0,60.0,0.9980,3.16,0.58,9.8]]))

