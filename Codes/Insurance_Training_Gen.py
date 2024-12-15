# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:51:01 2024

@author: asiva
"""

import pandas as pd
import numpy as np

df_train = pd.read_csv('train.csv')

df_train = df_train.iloc[:,1:]

X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]

"""
len(df_train) = 1200000
Columns with missing values:
    Age
    Annual Income
    Marital Status
    Number of Dependents
    Occupation
    health Score
    Previous Claims
    Vehicle Age
    Credit Score
    Insurance duration
    Customer Feedback
"""

temp = len(df_train)

from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer_median = SimpleImputer(missing_values = np.nan, strategy = "median")
imputer_mode = SimpleImputer(missing_values = np.nan, strategy = "most_frequent")

from sklearn.impute import KNNImputer
imputer_KNN = KNNImputer(missing_values = np.nan, n_neighbors = 5)

import matplotlib.pyplot as plt

ls_missing_values = []
for columns in df_train.columns:
    if(df_train[columns].isna().sum() != 0):
        ls_missing_values.append(columns)
        
for cols in ls_missing_values:
    if(isinstance(df_train[cols][0],str) is False):
        plt.hist(df_train[cols])
        plt.xlabel(f'{cols}')
        plt.ylabel('frequency')
        plt.title(f'Distribution of {cols}')
        plt.plot()
        plt.show()
        
        print(f'{cols}: {df_train[cols].skew()}')
   
        
d_na_col_types = {}
for cols in ls_missing_values:
    d_na_col_types[cols] = type(df_train[cols][0])
    
for cols in df_train.columns:
    print(cols)
    print(len(df_train[cols].unique()))

for cols in ls_missing_values:
    if(d_na_col_types[cols] != str and len(df_train[cols].unique()) > 10):
        if(abs(df_train[cols].skew()) > 0.5):
            temp = df_train[cols].values
            temp = temp.reshape(-1,1)
            temp = imputer_median.fit_transform(temp)
            df_train[cols] = temp
            
        else:
            temp = df_train[cols].values
            temp = temp.reshape(-1,1)
            temp = imputer_mean.fit_transform(temp)
            df_train[cols] = temp
            

new_temp_ls = []
for cols in ls_missing_values:
    if(df_train[cols].isna().sum() != 0):
        new_temp_ls.append(cols)
        
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()

temp = df_train['Marital Status'].values
temp = temp.reshape(-1,1)

temp = encoder.fit_transform(temp)
df_train['Marital Status'] = temp

temp = df_train['Occupation'].values
temp = temp.reshape(-1,1)

temp = encoder.fit_transform(temp)
df_train['Occupation'] = temp

temp = df_train['Customer Feedback'].values
temp = temp.reshape(-1,1)

temp = encoder.fit_transform(temp)
df_train['Customer Feedback'] = temp

for cols in df_train.columns:
    if(df_train[cols].isna().sum() != 0):
        temp =  df_train[cols].values
        temp = temp.reshape(-1,1)
        temp = imputer_mode.fit_transform(temp)
        df_train[cols] = temp


for cols in df_train.columns:
    if(isinstance(df_train[cols][0],str)):
        temp = df_train[cols].values
        temp = temp.reshape(-1,1)
        temp = encoder.fit_transform(temp)
        df_train[cols] = temp
        
df_train.to_csv('Preprocessed_Insurance_Data_New.csv')