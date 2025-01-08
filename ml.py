
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


X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    x_train_scaled, y_train, test_size=0.2)


##for cross-validation
cv=5

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return model_name, mae, mse, rmse, r2


###########Linear regression####################
#Mean R^2 for Linear Regression: 0.5766031088601306

#model=LinearRegression()
    ##train test split
'''
model.fit(X_train_split, y_train_split)
y_pred=model.predict(X_test_split)
evaluate_model(model, X_test_split, y_test_split, "Linear Regression")
'''
    ##cross-validation
'''
linear_cv_scores=cross_val_score(model, x_train_scaled, y_train.values.ravel(), cv=cv, scoring='r2')
print(f"Linear Regression CV Scores: {linear_cv_scores}")
print(f"Mean R^2 for Linear Regression: {linear_cv_scores.mean()}")
'''


##############Lasso regression############
# Mean R^2 for Lasso Regression: 0.6001533481959017
#model_lasso_best_param= Lasso(alpha= 0.08) #calling the model with the best parameter found: 0.08

    ##train test split
'''
model_lasso_best_param.fit(X_train_split, y_train_split)
y_pred=model_lasso_best_param.predict(X_test_split)
evaluate_model(model_lasso_best_param, X_test_split, y_test_split, "Lasso regression with 0.08")
'''
        ##feature selection
'''
important_features = np.abs(model_lasso_best_param.coef_) > 0.01  
x_train_selected = X_train_split[:, important_features]

print(f"Shape after keeping important feature: {x_train_selected.shape}")

x_test_selected = X_test_split[:, important_features]
'''

    ##cross-validation
'''
scores = cross_val_score(model_lasso_best_param, X_train_split, y_train.values.ravel(), cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
mean_rmse = -scores.mean()
print(f"Mean RMSE: {mean_rmse}")
'''


#########Ridge regression############
#Mean R^2 for Ridge Regression: 0.585754748401208
# #finding the parameter
'''
alphas = np.arange(0.01, 39.01, 0.01)
results = []
for a in alphas:
    model_ridge = Ridge(alpha=a)
    model_ridge.fit(X_train_split, y_train_split)

    y_pred = model_ridge.predict(X_test_split)
    mse = mean_squared_error(y_test_split, y_pred)
    results.append((a, mse))

# Find the alpha with the minimum MSE
best_alpha, best_mse = min(results, key=lambda x: x[1])
print(f"Best alpha: {best_alpha}, Best MSE: {best_mse}")
'''

'''
model_ridge_best_param= Ridge(alpha= 39)
#cross-validation
ridge_cv_scores=cross_val_score(model_ridge_best_param, x_train_scaled, y_train.values.ravel(), cv=cv, scoring='r2')
print(f"Ridge Regression CV Scores: {ridge_cv_scores}")
print(f"Mean R^2 for Ridge Regression: {ridge_cv_scores.mean()}")
'''
#model_lasso_best_param.fit(x_train_scaled, y_train.values.ravel())
#y_pred=model_lasso_best_param.predict(x_test_scaled)


#####Random Forest Regressor##############
##too computational expensive
'''
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor( n_estimators=500, n_jobs=-1,random_state=0)
rf_cv_scores = cross_val_score(rf_model, x_train_scaled, y_train.values.ravel(), cv=3, scoring='r2')
print(f"Random Forest CV Scores: {rf_cv_scores}")
print(f"Mean R^2 for Random Forest: {rf_cv_scores.mean()}")
'''

########Support Vector Regression (SVR)
#Mean R^2 for SVR Regressor: 0.3578130997955859
'''
from sklearn.svm import SVR

svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
# Cross-validation for the stacking regressor

stacking_cv_scores = cross_val_score(svr_model, x_train_scaled, y_train.values.ravel(), cv=3, scoring='r2', n_jobs=-1)
print(f"Stacking Regressor CV Scores: {stacking_cv_scores}")
print(f"Mean R^2 for Stacking Regressor: {stacking_cv_scores.mean()}")
'''
#use train test split
'''
svr_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_split = svr_model.predict(X_test_split)

# Evaluate the model
mae = mean_absolute_error(y_test_split, y_pred_split)
mse = mean_squared_error(y_test_split, y_pred_split)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split, y_pred_split)

print(f"Stacking Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")'''

#############Gradient Boosting Regressor
##too computational expensive
'''
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
# Cross-validation for the stacking regressor

gbr_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_split = gbr_model.predict(X_test_split)

# Evaluate the model
mae = mean_absolute_error(y_test_split, y_pred_split)
mse = mean_squared_error(y_test_split, y_pred_split)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split, y_pred_split)

print(f"Stacking Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
'''

##K-Nearest Neighbors (KNN)
##highly innacurate
'''
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_split = knn_model.predict(X_test_split)

# Evaluate the model
mae = mean_absolute_error(y_test_split, y_pred_split)
mse = mean_squared_error(y_test_split, y_pred_split)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split, y_pred_split)

print(f"Stacking Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
'''



#######ElasticNet
# Mean R^2 for ElasticNet Regressor: 0.5745607249738547
#Mean RMSE: 2.1588707106477916

#elasticnet_model = ElasticNet(alpha=0.08, l1_ratio=0.01) #good score
    ##train test split
'''
elasticnet_model.fit(X_train_split, y_train_split.values.ravel())
y_pred_train = elasticnet_model.predict(X_test_split)

print(f"Size y predicted: {y_pred_train.shape}")
print(f"Size y actual: {y_test_split.shape}")
'''
    ##cross-validation
'''
scores = cross_val_score(elasticnet_model, x_train_scaled, y_train.values.ravel(), cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
mean_rmse = -scores.mean()
print(f"Mean RMSE: {mean_rmse}")
'''





# Evaluate the model
'''
mae = mean_absolute_error(y_test_split, y_pred_train)
mse = mean_squared_error(y_test_split, y_pred_train)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split, y_pred_train)

print(f"ElasticNet Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
'''

####plot

'''
plt.figure(figsize=(8, 8))

plt.scatter(y_test_split, y_pred_train, color='#99CCFF', alpha=0.6, label='Predicted Points (y-axis)')
plt.plot(y_test_split, y_test_split, color='#000066',  label='Actual Points (x-axis)')

plt.title("Regressor: Actual vs Predicted")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

'''



#submission
#####################################

'''

y_pred_final=model_lasso_best_param.predict(x_test_scaled)


submission_df = pd.DataFrame({
    'participant_id': x_test['participant_id'],  
    'age': y_pred_final  # Predicted ages
})
if len(submission_df) != 474:
    print(f"Expected 474 rows, but got {len(submission_df)}")
submission_file_path = os.path.join(project_dir, 'submission_elasticnet3.csv')
submission_df.to_csv(submission_file_path, index=False)
'''



