import math
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

project_dir = os.path.dirname(__file__)

x_train_input_path = os.path.join(project_dir, 'output_resources/x_train.csv')
y_train_input_path = os.path.join(project_dir, 'output_resources/y_train.csv')
x_test_input_path = os.path.join(project_dir, 'output_resources/x_test.csv')

x_train = pd.read_csv(x_train_input_path)
y_train = pd.read_csv(y_train_input_path)
x_test = pd.read_csv(x_test_input_path)
print("Loaded")

df = x_train.select_dtypes(include=["number"])
df_test = x_test.select_dtypes(include=["number"])

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df)
x_test_scaled = scaler.transform(df_test)

# split into train-test
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    x_train_scaled, y_train, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return model_name, mae, mse, rmse, r2

models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.08),
    "Ridge Regression": Ridge(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.08, l1_ratio=0.01)
}

results = []

for model_name, model in models.items():
    model.fit(X_train_split, y_train_split.values.ravel())
    result = evaluate_model(model, X_test_split, y_test_split, model_name)
    results.append(result)

results_df = pd.DataFrame(results, columns=["Model", "MAE", "MSE", "RMSE", "R²"])
print(results_df)

results_df.set_index('Model')[['MAE', 'MSE', 'RMSE', 'R²']].plot(kind='bar', figsize=(10, 6))
plt.title("Model Comparison")
plt.ylabel("Scores")
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
