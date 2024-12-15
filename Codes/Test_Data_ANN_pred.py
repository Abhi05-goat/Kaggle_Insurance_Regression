# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:22:28 2024

@author: asiva
"""

f_test = pd.read_csv('Preprocessed_Test_Data.csv')
X_test = df_test.iloc[:,1:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_test = scaler.fit_transform(X_test)

y_predict = ann.predict(X_test)

Target_column = pd.DataFrame(y_predict, columns = ['Premium Amount'])

ID_column = pd.DataFrame(temp_id, columns = ['id'])

df_final = pd.concat([ID_column,Target_column], axis = 1)

df_final.to_csv("Kaggle_Final_Insurance_Submission.csv")