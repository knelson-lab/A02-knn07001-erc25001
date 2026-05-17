# Step 1: import the necessary libraries and reading the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

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
mlp = MLPRegressor(hidden_layer_sizes = (100, 50), 
                   activation = 'relu',
                   max_iter = 500,  
                   random_state = 42, 
                   early_stopping = True)
mlp.fit(X_train_scaled, y_train)

# Step 5: Evaluate the model on the validation set

y_val_pred = mlp.predict(X_val_scaled)
print("Validation MSE:", mean_squared_error(y_val, y_val_pred))
print("Validation R2:", r2_score(y_val, y_val_pred))
# -----------------------------
# Step 5: Train predictions + plot (PR #3)
# -----------------------------
train_pred = mlp.predict(X_train_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(y_train, train_pred, alpha=0.3)
plt.xlabel("Actual Train Values")
plt.ylabel("Predicted Train Values")
plt.title("MLPRegressor — Training Predictions")
plt.grid(True)
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "r--",
    linewidth=2
)
plt.tight_layout()
plt.savefig("../figs/train_predictions.png")
plt.close()
