import math
import pandas as pd
import os
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

project_dir = os.path.dirname(__file__)

x_train_input_path = os.path.join(project_dir, 'output_resources/x_train.csv')
y_train_input_path = os.path.join(project_dir, 'output_resources/y_train.csv')
x_test_input_path = os.path.join(project_dir, 'output_resources/x_test.csv')

x_train = pd.read_csv(x_train_input_path)
y_train = pd.read_csv(y_train_input_path)
x_test = pd.read_csv(x_test_input_path)
print("Loaded")

# Selecting only numerical columns
df = x_train.select_dtypes(include=["number"])
df_test = x_test.select_dtypes(include=["number"])

# Scaling the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df)
x_test_scaled = scaler.transform(df_test)

# Train-test split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
      x_train_scaled, y_train, test_size=0.2, random_state=42)

####### Lasso Regressor

lasso_model = Lasso(alpha=0.08, max_iter=10000, random_state=42)

lasso_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_train = lasso_model.predict(X_test_split)

# Print the sizes of the predictions and actual values
print(f"Size y predicted: {y_pred_train.shape}")
print(f"Size y actual: {y_test_split.shape}")

y_test_split_flat = y_test_split.values.ravel()  # Flatten y_test_split to (221,)

# Evaluate the model
mae = mean_absolute_error(y_test_split_flat, y_pred_train)
mse = mean_squared_error(y_test_split_flat, y_pred_train)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split_flat, y_pred_train)

print(f"Lasso Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test_split_flat, y_pred_train, color='#99CCFF', alpha=0.6, label='Predicted Points (y-axis)')
plt.plot(y_test_split_flat, y_test_split_flat, color='#000066', label='Actual Points (x-axis)')
plt.title("Lasso Regressor: Actual vs Predicted")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Final Predictions for the submission
y_pred_final = lasso_model.predict(x_test_scaled)
print(f"Size y predicted final: {y_pred_final.shape}")

# Prepare submission DataFrame
submission_df = pd.DataFrame({
    'participant_id': x_test['participant_id'],  # Get participant IDs from x_test
    'age': y_pred_final  # Predicted ages
})

# Check if the number of rows is correct (Expected 474 rows)
if len(submission_df) != 474:
    print(f"Expected 474 rows, but got {len(submission_df)}")

# Save the submission file
submission_file_path = os.path.join(project_dir, 'submission_lasso.csv')
submission_df.to_csv(submission_file_path, index=False)
