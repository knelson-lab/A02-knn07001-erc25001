# Step 1: import the necessary libraries and reading the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import networkx as nx

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
# Step 6: Train predictions + plot (PR #3)
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
plt.savefig("./figs/train_predictions.png")
plt.close()


# Step 7: Plot epoch vs. validation score
val_scores = mlp.validation_scores_

plt.figure(figsize = (8, 5))
plt.plot(val_scores, marker = "o")
plt.axhline(
    mlp.best_validation_score_,
    linestyle = "--",
    label=f"Best val score = {mlp.best_validation_score_:.3f}"
)
plt.xlabel("Epoch")
plt.ylabel("Validation score (R²)")
plt.title("Validation Score vs Epoch (sklearn MLPRegressor)")
plt.legend()
plt.grid(True)
plt.savefig("./figs/epoch_vs.validation.png")

# Step 8: Evaluate the model on the test set
y_pred_val   = mlp.predict(X_val_scaled)
y_pred_test  = mlp.predict(X_test_scaled)

# Step 9: Evaluate metrics for model fit 
def metrics_row(name, y_true, y_pred):
    return {
        "split": name,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }

metrics_df = pd.DataFrame([
    metrics_row("train", y_train, train_pred),
    metrics_row("val",   y_val,   y_pred_val),
    metrics_row("test",  y_test,  y_pred_test),
])

print("=== Metrics (defaults) ===")
print(metrics_df.to_string(index = False))

# Step 10: Plot actual vs predicted values for the test set
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel("Actual Test Values")
plt.ylabel("Predicted Test Values")
plt.title("MLPRegressor — Test Predictions")
plt.grid(True)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    linewidth=2
)
plt.tight_layout()
plt.savefig("./figs/test_predictions.png")
plt.show()
