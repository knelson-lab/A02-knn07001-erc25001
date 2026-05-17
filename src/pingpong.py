# Step 1: import the necessary libraries and reading the data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
df.info()
print(df.describe())

# Step2: Create train/test split
# Make MedHouseVal the response variable and the rest of the columns the features
X = df.drop(columns = ["MedHouseVal"])
y = df["MedHouseVal"]

# Allocate 20% of data for training
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Allocate 40% of data for validation and 40% for testing
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

# Step 3: Scale data for model training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Step 4: Train a simple model with limited hyperparameters aside from early stopping
mlp = MLPRegressor(early_stopping = True)
mlp.fit(X_train_scaled, y_train)
