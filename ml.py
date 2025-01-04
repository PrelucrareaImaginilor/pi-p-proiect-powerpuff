import math

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
x_test_scaled = scaler.fit_transform(df_test)
#y_train_final = scaler.fit_transform(y_train)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
    x_train_scaled, y_train, test_size=0.2)

###########Linear regression####################
'''
model=LinearRegression()
model.fit(X_train_split, y_train_split)
y_pred=abs(model.predict(X_test_split))
print("Linear Regression")
'''
#print(y_pred)
'''
# # Female only data frame
# female_X_train = x_train_final[x_train_final['sex_Male'] == 0]
# female_X_test = x_test_final[x_test_final['sex_Male'] == 0]
# #Female age predictions
# y_train_female = y_train[x_train['sex_Male'] == 0]['age']
#
# # Male only data frame
# male_X_train = x_train_final[x_train_final['sex_Male'] ==  1]
# male_X_test = x_test_final[x_test_final['sex_Male'] ==  1]
# #Male age predictions
# y_train_male = y_train[x_test['sex_Male'] == 1]['age']
#
# model.fit(female_X_train, y_train_female)
# model.fit(male_X_train, y_train_male)
#
# scores_male = cross_val_score(model, male_X_train, y_train_male, cv=5)  # 5-fold cross-validation
# # print("Cross-validated scores for Male:", scores_male)
# scores_female = cross_val_score(model, female_X_train, y_train_female, cv=5)  # 5-fold cross-validation
# # print("Cross-validated scores for Female:", scores_female)
# male_avg_score = scores_male.mean()
# female_avg_score = scores_female.mean()
# print("Average cross-validated score for Male:", male_avg_score)
# print("Average cross-validated score for Female:", female_avg_score)
'''


##############Lasso regression############
'''
#params = {"alpha": np.logspace(-3, 1, 50)}


# Number of Folds and adding the random state for replication
# kf=KFold(n_splits=5,shuffle=True, random_state=42)
# # first lasso model
# model_lasso1 = Lasso(random_state=0)
# #define parameters dictionary
# alpha_range=np.arange(0.07, 2, 0.1)
# param={'alpha': alpha_range}
# #score base on r2
# grid_search_r2 = GridSearchCV(model_lasso1, param_grid=param, cv=5)
# grid_search_r2.fit(X_train_split, y_train_split)
# #scor base on mae
# grid_search_mae=GridSearchCV(model_lasso1, param_grid=param, scoring='neg_mean_absolute_error', cv=5)
# grid_search_mae.fit(X_train_split, y_train_split)

#results
# print('Scoring R2')
# print('Best R2 score   : ', grid_search_r2.best_score_)
# print('Best parameters : ', grid_search_r2.best_params_)
# print()
#
#
# # results
# print('Scoring MAE')
# print('Best MAE score  : ', abs(grid_search_mae.best_score_))
# print('Best parameters : ', grid_search_mae.best_params_)

# Loaded
# Scoring R2
# Best R2 score   :  0.6001533481959017
# Best parameters :  {'alpha': np.float64(0.08)}
#
# Scoring MAE
# Best MAE score  :  1.6199736242962175
# Best parameters :  {'alpha': np.float64(0.08)}
# model_lasso1.fit(x_train_final, y_train)
# print("first lasso model prediction")
# y_pred=model_lasso1.predict(x_test_final)
# print(y_pred)
#print("Best Params {}".format(model_lasso1.best_params_)) #get best alpha value
'''

# calling the model with the best parameter
model=Lasso()
#model_lasso_best_param= Lasso(alpha= 0.07196856730011521) #this is the alpha we got from fitting the model!

model_lasso_best_param= Lasso(alpha= 0.08) #0.08 r2: 0.6139
model_lasso_best_param.fit(X_train_split, y_train_split)
y_pred=model_lasso_best_param.predict(X_test_split)
print("Lasso Regression")



mae=mean_absolute_error(y_test_split, y_pred)
mse=mean_squared_error(y_test_split, y_pred)
rmse=math.sqrt(mse)
r2=r2_score(y_test_split, y_pred)

print(f"Validation Mean Absolute Error (MAE): {mae}")
print(f"Validation Mean Squared Error (MSE): {mse}")
print(f"Validation Root Mean Squared Error (RMSE): {rmse}")
print(f"Validation R^2 Score: {r2}")

y_pred_final=model_lasso_best_param.predict(x_test_scaled)


submission_df = pd.DataFrame({
    'participant_id': x_test['participant_id'],  # Get participant IDs from x_test
    'age': y_pred_final  # Predicted ages
})
if len(submission_df) != 474:
    print(f"Expected 474 rows, but got {len(submission_df)}")
submission_file_path = os.path.join(project_dir, 'submission_lasso_0.08_iter.csv')
submission_df.to_csv(submission_file_path, index=False)


# # Create a DataFrame of coefficients
# names=df.columns
# x_train_final_df = pd.DataFrame(x_train_final, columns=names)
# # Using np.abs() to make coefficients positive.
# lasso1_coef = np.abs(lasso1.coef_)
# coefficients_df = pd.DataFrame({'Feature': names, 'Coefficient': lasso1_coef})
# coefficients = coefficients_df[coefficients_df['Coefficient'] > 0.10]
# sorted_coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
# print("\nCoefficients greater than 0.10 :")
# print(sorted_coefficients)

