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

lasso_model = Lasso(alpha=0.086, max_iter=10000, random_state=42)

lasso_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_train = lasso_model.predict(X_test_split)

# Print the sizes of the predictions and actual values
print(f"Size y predicted: {y_pred_train.shape}")
print(f"Size y actual: {y_test_split.shape}")

y_test_split_flat = y_test_split.values.ravel()

mae = mean_absolute_error(y_test_split_flat, y_pred_train)
mse = mean_squared_error(y_test_split_flat, y_pred_train)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split_flat, y_pred_train)

print(f"Lasso Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")


plt.figure(figsize=(8, 8))
plt.scatter(y_test_split_flat, y_pred_train, color='#99CCFF', alpha=0.6, label='Predicted Points (y-axis)')
plt.plot(y_test_split_flat, y_test_split_flat, color='#000066', label='Actual Points (x-axis)')
plt.title("Lasso Regressor: Actual vs Predicted")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

y_pred_final = lasso_model.predict(x_test_scaled)
print(f"Size y predicted final: {y_pred_final.shape}")

submission_df = pd.DataFrame({
    'participant_id': x_test['participant_id'],  # Get participant IDs from x_test
    'age': y_pred_final  # Predicted ages
})


if len(submission_df) != 474:
    print(f"Expected 474 rows, but got {len(submission_df)}")


submission_file_path = os.path.join(project_dir, 'submission_lasso.csv')
submission_df.to_csv(submission_file_path, index=False)

#checking for another alpha (not a computationally expensive method)

# alphas = [0.078, 0.079, 0.081, 0.082, 0.083, 0.086]
#
# results = {'alpha': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R²': []}
#
# for alpha in alphas:
#     lasso_model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
#
#     lasso_model.fit(X_train_split, y_train_split.values.ravel())
#     y_pred_train = lasso_model.predict(X_test_split)
#
#     mae = mean_absolute_error(y_test_split.values.ravel(), y_pred_train)
#     mse = mean_squared_error(y_test_split.values.ravel(), y_pred_train)
#     rmse = math.sqrt(mse)
#     r2 = r2_score(y_test_split.values.ravel(), y_pred_train)
#
#     results['alpha'].append(alpha)
#     results['MAE'].append(mae)
#     results['MSE'].append(mse)
#     results['RMSE'].append(rmse)
#     results['R²'].append(r2)
#
# results_df = pd.DataFrame(results)
#
# plt.figure(figsize=(10, 6))
#
# # mae
# plt.subplot(2, 2, 1)
# plt.plot(results_df['alpha'], results_df['MAE'], marker='o', color='blue', label='MAE')
# plt.title('Mean Absolute Error (MAE) vs Alpha')
# plt.xlabel('Alpha')
# plt.ylabel('MAE')
# plt.grid(True)
#
# # mse
# plt.subplot(2, 2, 2)
# plt.plot(results_df['alpha'], results_df['MSE'], marker='o', color='green', label='MSE')
# plt.title('Mean Squared Error (MSE) vs Alpha')
# plt.xlabel('Alpha')
# plt.ylabel('MSE')
# plt.grid(True)
#
# # rmse
# plt.subplot(2, 2, 3)
# plt.plot(results_df['alpha'], results_df['RMSE'], marker='o', color='red', label='RMSE')
# plt.title('Root Mean Squared Error (RMSE) vs Alpha')
# plt.xlabel('Alpha')
# plt.ylabel('RMSE')
# plt.grid(True)
#
# # r2
# plt.subplot(2, 2, 4)
# plt.plot(results_df['alpha'], results_df['R²'], marker='o', color='purple', label='R²')
# plt.title('R² vs Alpha')
# plt.xlabel('Alpha')
# plt.ylabel('R²')
# plt.grid(True)
#
# # Show the plot
# plt.tight_layout()
# plt.show()
#
# # Display results DataFrame
# print(results_df)
# #
# #    alpha       MAE       MSE      RMSE        R²
# # 0   0.08  1.635893  3.973403  1.993340  0.583410
# # 1   0.10  1.643521  3.943285  1.985771  0.586568
# # 2   0.05  1.652710  4.146423  2.036277  0.565270
# # 3   0.20  1.662652  4.072131  2.017952  0.573059
# # 4   0.50  1.986632  5.702899  2.388074  0.402081
# # 5   1.00  2.355087  7.773542  2.788107  0.184985
