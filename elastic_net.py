import math

import pandas as pd
import os
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNet
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from unicodedata import normalize
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from ml import z_scores

project_dir = os.path.dirname(__file__)

x_train_input_path = os.path.join(project_dir, 'output_resources/x_train.csv')
y_train_input_path = os.path.join(project_dir, 'output_resources/y_train.csv')
x_test_input_path = os.path.join(project_dir, 'output_resources/x_test.csv')

x_train = pd.read_csv(x_train_input_path)
y_train = pd.read_csv(y_train_input_path)
x_test = pd.read_csv(x_test_input_path)
print("Loaded")

#selectign only numerical columns
df = x_train.select_dtypes(include=["number"])
df_test=x_test.select_dtypes(include=["number"])

#scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df)
x_test_scaled = scaler.transform(df_test)

#remove outliers using z_score
z_scores=np.abs(x_train_scaled)
z_threshold=3
outliers=(z_scores>z_threshold).any(axis=1)

df_cleaned = df[~outliers]
df_test_cleaned = df_test[~outliers]

x_train_scaled_cleaned = scaler.fit_transform(df_cleaned)
x_test_scaled_cleaned = scaler.transform(df_test_cleaned)


X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
     x_train_scaled, y_train, test_size=0.2)

# X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
#     x_train_scaled_cleaned, y_train, test_size=0.2)

#######ElasticNet

# Mean R^2 for ElasticNet Regressor: 0.5745607249738547
#Mean RMSE: 2.1588707106477916

elasticnet_model = ElasticNet(alpha=0.08, l1_ratio=0.01) #good score
    ##train test split
elasticnet_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_train = abs(elasticnet_model.predict(X_test_split))
    ##cross-validation

print(f"Size y predicted: {y_pred_train.shape}")
print(f"Size y actual: {y_test_split.shape}")


y_test_split_flat= y_test_split.values.ravel()  # Flatten y_test_split to (221,)


# Evaluate the model
mae = mean_absolute_error(y_test_split_flat, y_pred_train)
mse = mean_squared_error(y_test_split_flat, y_pred_train)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split_flat, y_pred_train)

print(f"ElasticNet Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

plt.figure(figsize=(8, 8))

plt.scatter(y_test_split_flat, y_pred_train, color='#99CCFF', alpha=0.6, label='Predicted Points (y-axis)')
plt.plot(y_test_split_flat, y_test_split_flat, color='#000066',  label='Actual Points (x-axis)')


plt.title("ElasticNet Regressor: Actual vs Predicted")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

#submission
#####################################

y_pred_final=elasticnet_model.predict(x_test_scaled)


submission_df = pd.DataFrame({
    'participant_id': x_test['participant_id'],  # Get participant IDs from x_test
    'age': y_pred_final  # Predicted ages
})
if len(submission_df) != 474:
    print(f"Expected 474 rows, but got {len(submission_df)}")
submission_file_path = os.path.join(project_dir, 'submission_elasticnet.csv')
submission_df.to_csv(submission_file_path, index=False)



