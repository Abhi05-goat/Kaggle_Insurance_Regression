# Kaggle_Insurance_Regression
Training and implementing an ANN regression to predict the insurance premiums based on the set of features given in the Kaggle competition dataset.
Project: Insurance Premium Prediction
This project predicts the premium amount for insurance based on various customer attributes. The dataset contains information about customers such as age, income, marital status, and more. The model processes the data, handles missing values, applies feature encoding, and predicts the insurance premium using a neural network.

Project Structure
Data Preprocessing:

The data is loaded and missing values are handled using various imputation strategies.
Categorical features are encoded using ordinal encoding.
The data is scaled using StandardScaler for model training.
Model Training:

The preprocessed training data is used to train a neural network (ANN) to predict the insurance premium.
The model uses 2 hidden layers with ReLU activation and outputs a single prediction value.
Test Data Preprocessing and Prediction:

Similar preprocessing steps are applied to the test data.
The trained model is used to predict the insurance premiums for the test data.
Final Output:

The predictions are saved in a CSV file containing the predicted premium amounts along with their respective IDs.
Dependencies
pandas: For data manipulation and handling CSV files.
numpy: For numerical operations and handling arrays.
matplotlib: For visualizing missing values and distributions.
sklearn: For preprocessing, handling missing values, and scaling the data.
tensorflow: For building and training the neural network model.
