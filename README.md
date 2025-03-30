# Kaggle Sticker Sales Prediction

## Overview
This repository contains an end-to-end solution for predicting the number of stickers sold using **XGBoost Regression**. The dataset is preprocessed, encoded, and trained on an **XGBoost model** to generate predictions for a Kaggle competition.

## Features
- Data preprocessing and cleaning (handling missing values, encoding categorical features, extracting date-based features).
- Feature engineering for better model performance.
- **XGBoost Regressor** implementation for training and predictions.
- Model evaluation with **Mean Squared Error (MSE)**.
- Final prediction submission generation for Kaggle.

## Dataset
- `train.csv`: Contains historical sticker sales data.
- `test.csv`: Dataset for generating predictions.
- `Preprocessed_Insurance_Data_New.csv`: Used for an additional deep learning model.
- `Preprocessed_Test_Data.csv`: Processed test dataset.

## Installation
Ensure you have the necessary libraries installed:

```bash
pip install numpy pandas matplotlib xgboost tensorflow scikit-learn
