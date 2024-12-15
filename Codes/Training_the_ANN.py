# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:53:02 2024

@author: asiva
"""

import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv('Preprocessed_Insurance_Data_New.csv')
df = df.iloc[:,1:]

X_train = df.iloc[:,:-1]
y_train = df.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer='adam',        # Adaptive moment estimation optimizer
              loss='mse',              # Mean Squared Error loss for regression
              metrics=['mae'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 50)