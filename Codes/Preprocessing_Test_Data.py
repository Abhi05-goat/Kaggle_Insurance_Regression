# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:53:04 2024

@author: asiva
"""

import pandas as pd
import numpy as np

df_test = pd.read_csv('test.csv')
temp_id = df_test.iloc[:,0]


df_test = df_test.iloc[:, 1:]  # Drop ID column

"""
len(df_test) = <actual length of test dataset>
Columns with missing values:
    Age
    Annual Income
    Marital Status
    Number of Dependents
    Occupation
    Health Score
    Previous Claims
    Vehicle Age
    Credit Score
    Insurance Duration
    Customer Feedback
"""

temp = len(df_test)

from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer_median = SimpleImputer(missing_values=np.nan, strategy="median")
imputer_mode = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

from sklearn.impute import KNNImputer
imputer_KNN = KNNImputer(missing_values=np.nan, n_neighbors=5)

import matplotlib.pyplot as plt

ls_missing_values = []
for columns in df_test.columns:
    if df_test[columns].isna().sum() != 0:
        ls_missing_values.append(columns)

d_na_col_types = {}
for cols in ls_missing_values:
    d_na_col_types[cols] = type(df_test[cols][0])

for cols in df_test.columns:
    print(cols)
    print(len(df_test[cols].unique()))

for cols in ls_missing_values:
    if (d_na_col_types[cols] != str and len(df_test[cols].unique()) > 10):
        if abs(df_test[cols].skew()) > 0.5:
            temp = df_test[cols].values
            temp = temp.reshape(-1, 1)
            temp = imputer_median.fit_transform(temp)
            df_test[cols] = temp

        else:
            temp = df_test[cols].values
            temp = temp.reshape(-1, 1)
            temp = imputer_mean.fit_transform(temp)
            df_test[cols] = temp

new_temp_ls = []
for cols in ls_missing_values:
    if df_test[cols].isna().sum() != 0:
        new_temp_ls.append(cols)

from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()

temp = df_test['Marital Status'].values
temp = temp.reshape(-1, 1)

temp = encoder.fit_transform(temp)
df_test['Marital Status'] = temp

temp = df_test['Occupation'].values
temp = temp.reshape(-1, 1)

temp = encoder.fit_transform(temp)
df_test['Occupation'] = temp

temp = df_test['Customer Feedback'].values
temp = temp.reshape(-1, 1)

temp = encoder.fit_transform(temp)
df_test['Customer Feedback'] = temp

for cols in df_test.columns:
    if df_test[cols].isna().sum() != 0:
        temp = df_test[cols].values
        temp = temp.reshape(-1, 1)
        temp = imputer_mode.fit_transform(temp)
        df_test[cols] = temp

for cols in df_test.columns:
    if isinstance(df_test[cols][0], str):
        temp = df_test[cols].values
        temp = temp.reshape(-1, 1)
        temp = encoder.fit_transform(temp)
        df_test[cols] = temp

df_test.to_csv('Preprocessed_Test_Data.csv')