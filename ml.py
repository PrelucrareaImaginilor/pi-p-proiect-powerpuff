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
#y_train_scaled = scaler.fit_transform(y_train)

# print(f"Mean after scaling: {np.mean(x_train_scaled, axis=0)}")
# print(f"Variance after scaling: {np.var(x_train_scaled, axis=0)}")
# Mean after scaling: [ 0.00000000e+00  3.70074342e-17  5.76028758e-16 ... -4.98795852e-17
#   5.79246795e-17  6.61306758e-16]
# Variance after scaling: [0. 1. 1. ... 1. 1. 1.]
non_zero_variance_columns = np.var(x_train_scaled, axis=0) != 0

x_train_cleaned = x_train_scaled[:, non_zero_variance_columns]
x_test_cleaned = x_test_scaled[:, non_zero_variance_columns]



from scipy.stats import zscore

z_scores = np.abs(zscore(x_train_cleaned))
outliers = (z_scores > 3).all(axis=1)
x_train_no_outliers = x_train_cleaned[~outliers]



z_scores_test = np.abs(zscore(x_test_cleaned))
outliers_test = (z_scores_test > 3).all(axis=1)  # Flag rows with outliers
x_test_no_outliers = x_test_cleaned[~outliers_test]



X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    x_train_no_outliers, y_train, test_size=0.2)




##for cross-validation
cv=5

def evaluate_model(model, xx_test, yy_test, model_name):
    y_pred_final = model.predict(xx_test)
    mae = mean_absolute_error(yy_test, y_pred_final)
    mse = mean_squared_error(yy_test, y_pred_final)
    rmse = math.sqrt(mse)
    r2 = r2_score(yy_test, y_pred_final)

    print(f"{model_name} Results:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    return mae, mse, rmse, r2


###########Linear regression####################
#Mean R^2 for Linear Regression: 0.5766031088601306
#model=LinearRegression()
'''
model.fit(X_train_split, y_train_split)
y_pred=abs(model.predict(X_test_split))
evaluate_model(model, X_test_split, y_test_split, "Linear Regression")
'''
'''
linear_cv_scores=cross_val_score(model, x_train_scaled, y_train.values.ravel(), cv=cv, scoring='r2')
print(f"Linear Regression CV Scores: {linear_cv_scores}")
print(f"Mean R^2 for Linear Regression: {linear_cv_scores.mean()}")
'''


##############Lasso regression############
# Mean R^2 for Lasso Regression: 0.6001533481959017
model_lasso_best_param= Lasso(alpha= 0.08) #calling the model with the best parameter found: 0.08

#manual splitting dataset
'''
model_lasso_best_param.fit(X_train_split, y_train_split)
y_pred=model_lasso_best_param.predict(X_test_split)
evaluate_model(model_lasso_best_param, X_test_split, y_test_split, "Lasso regression with 0.08")
'''
#cross-validation

model_lasso_best_param.fit(X_train_split, y_train.values.ravel())
y_pred=model_lasso_best_param.predict(X_test_split)

scores = cross_val_score(model_lasso_best_param, X_train_split, y_train.values.ravel(), cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
mean_rmse = -scores.mean()
print(f"Mean RMSE: {mean_rmse}")

important_features = np.abs(model_lasso_best_param.coef_) > 0.01  # Adjust threshold as needed
x_train_selected = X_train_split[:, important_features]

print(f"Shape after keeping important feature: {x_train_selected.shape}")

x_test_selected = X_test_split[:, important_features]




#########Ridge regression############
#alpha:39
    #Mean R^2 for Ridge Regression: 0.5770820748027898
#gridSearch alpha
    #Mean R^2 for Ridge Regression: 0.585754748401208
# #finding the parameter
'''

alphas = range(1,40)

for a in alphas:
    model_ridge=Ridge(alpha=a)
    model_ridge.fit(X_train_split, y_train_split)

regr_cv=RidgeCV(alphas=range(1,40))
model_cv=regr_cv.fit(X_train_split, y_train_split)

print(model_cv.alpha_)#39
'''
'''
ridge_params = {'alpha': np.logspace(-3, 3, 50)}
ridge_model = GridSearchCV(Ridge(), param_grid=ridge_params, scoring='neg_mean_squared_error', cv=5)
ridge_cv_scores=cross_val_score(ridge_model, x_train_scaled, y_train.values.ravel(), cv=cv, scoring='r2')
print(f"Ridge Regression CV Scores: {ridge_cv_scores}")
print(f"Mean R^2 for Ridge Regression: {ridge_cv_scores.mean()}")

ridge_model.fit(X_train_split, y_train_split)
best_ridge = ridge_model.best_estimator_
print(f"Best Ridge Alpha: {ridge_model.best_params_['alpha']}")
evaluate_model(best_ridge, X_test_split, y_test_split, "Ridge Regression")
'''

'''
model_ridge_best_param= Ridge(alpha= 39) #calling the model with the best parameter found: 0.08
#cross-validation
ridge_cv_scores=cross_val_score(model_ridge_best_param, x_train_scaled, y_train.values.ravel(), cv=cv, scoring='r2')
print(f"Ridge Regression CV Scores: {ridge_cv_scores}")
print(f"Mean R^2 for Ridge Regression: {ridge_cv_scores.mean()}")
'''
#model_lasso_best_param.fit(x_train_scaled, y_train.values.ravel())
#y_pred=model_lasso_best_param.predict(x_test_scaled)


#####Random Forest Regressor##############
'''
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor( n_estimators=500, n_jobs=-1,random_state=0)
print("done")
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
'''
#too highly computational
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
'''
#highly innacurate
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
elasticnet_model = ElasticNet(alpha=0.08, l1_ratio=0.01)
elasticnet_model.fit(x_train_selected, y_train)
##cross-validation
'''
scores = cross_val_score(elasticnet_model, x_train_scaled, y_train.values.ravel(), cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
mean_rmse = -scores.mean()
print(f"Mean RMSE: {mean_rmse}")
'''

y_pred_train = elasticnet_model.predict(x_train_selected)

# Evaluate the model
mae = mean_absolute_error(y_test_split, y_pred_train)
mse = mean_squared_error(y_test_split, y_pred_train)
rmse = math.sqrt(mse)
r2 = r2_score(y_test_split, y_pred_train)

print(f"Stacking Regressor Test Results:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")













#submission
#####################################

'''
y_pred_final=model_lasso_best_param.predict(x_test_scaled)


submission_df = pd.DataFrame({
    'participant_id': x_test['participant_id'],  # Get participant IDs from x_test
    'age': y_pred_final  # Predicted ages
})
if len(submission_df) != 474:
    print(f"Expected 474 rows, but got {len(submission_df)}")
submission_file_path = os.path.join(project_dir, 'submission_elasticnet3.csv')
submission_df.to_csv(submission_file_path, index=False)
'''


