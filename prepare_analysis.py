import os
import pandas as pd
from process import process_folder
#from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np

project_dir = os.path.dirname(__file__)

output_file_train = 'output_resources/train_output.tsv'
output_file_test = 'output_resources/test_output.tsv'
output_file_train_path = os.path.join(project_dir, output_file_train)
output_file_test_path = os.path.join(project_dir, output_file_test)

metadata_train = 'resources/metadata/training_metadata.csv'
metadata_test = 'resources/metadata/test_metadata.csv'
metadata_train_path = os.path.join(project_dir, metadata_train)
metadata_test_path = os.path.join(project_dir, metadata_test)

if os.path.exists(output_file_train_path) and os.path.exists(output_file_test_path):
    train_data = pd.read_csv(output_file_train_path, sep='\t')
    test_data = pd.read_csv(output_file_test_path, sep='\t')
    #print("Nan values in train data:", pd.isna(test_data).sum().sum()) #Nan values in train data: 0
    #print("Nan values in test data:", pd.isna(train_data).sum().sum())  #Nan values in test data: 1188

train_metadata=pd.read_csv(metadata_train_path)
#print("Nan values in train metadata:", pd.isna(train_metadata).sum().sum()) #Nan values in train metadata: 500
test_metadata=pd.read_csv(metadata_test_path)
#print("Nan values in test metadata:", pd.isna(test_metadata).sum().sum())   #Nan values in test metadata: 230

# check for nan values per column in the DataFrame
nan_counts = train_metadata.isna().sum()
print("NaN counts per column:")
print(nan_counts)
#handling categorical data from metadata, no null values

#############1. handle categorical data
'''
####1.1.sex
gender_label=LabelEncoder() #assigns integers based on the alphabetical order of the unique categorical values
train_metadata['sex']=gender_label.fit_transform(train_metadata['sex'])
test_metadata['sex']=gender_label.fit_transform(test_metadata['sex'])

#categorical check values
categorical_cols = ['study_site', 'ethnicity', 'race', 'handedness', 'parent_1_education', 'parent_2_education']

# Check distinct values for each column in train_metadata
for column in categorical_cols:
    distinct_values_train = train_metadata[column].unique()
    print(f"Distinct values in 'train_metadata' for '{column}': {distinct_values_train}")

# Check distinct values for each column in test_metadata
for column in categorical_cols:
    distinct_values_test = test_metadata[column].unique()
    print(f"Distinct values in 'test_metadata' for '{column}': {distinct_values_test}")
####1.2.study_site
distinct_values = train_metadata['study_site'].unique()
print(f"Distinct values in 'study_site': {distinct_values}")
####1.3.ethnicity
####1.4.race
####1.5.handedness
####1.6.parent_1_education
####1.7.parent_2_education
'''
categorical_cols = ['sex', 'study_site', 'ethnicity', 'race', 'handedness', 'parent_1_education', 'parent_2_education']
#handle nan values for categorical
train_metadata.loc[:, categorical_cols] = train_metadata.loc[:, categorical_cols].fillna("Unknown")
test_metadata.loc[:, categorical_cols] = test_metadata.loc[:, categorical_cols].fillna("Unknown")
####handle nan values bmi
    #we group the data by sex and impute the bmi using the mean for each group
sex_group = train_metadata.groupby('sex')['bmi'].mean()
train_metadata['bmi'] = train_metadata['bmi'].fillna(train_metadata['sex'].map(sex_group))
test_metadata['bmi'] = test_metadata['bmi'].fillna(test_metadata['sex'].map(sex_group))
#check if any nan values left
nan_counts = test_metadata.isna().sum()
print("NaN counts per column:")
print(nan_counts)

# Convert categorical columns to strings to avoid mixed types
train_metadata.loc[:, categorical_cols] = train_metadata.loc[:, categorical_cols].astype(str)
test_metadata.loc[:, categorical_cols] = test_metadata.loc[:, categorical_cols].astype(str)
# Apply One-Hot Encoding to the categorical columns
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')  # drop='first' to avoid multicollinearity
# Fit the encoder on X_train and transform both X_train and X_test
train_metadata_encoded = encoder.fit_transform(train_metadata[categorical_cols])
test_metadata_encoded = encoder.transform(test_metadata[categorical_cols])

train_metadata_encoded_dense = train_metadata_encoded.todense()
test_metadata_encoded_dense = test_metadata_encoded.todense()

train_metadata_encoded_df = pd.DataFrame(train_metadata_encoded_dense, columns=encoder.get_feature_names_out(categorical_cols))
test_metadata_encoded_df = pd.DataFrame(test_metadata_encoded_dense, columns=encoder.get_feature_names_out(categorical_cols))

# Keep the original numerical columns (including 'bmi' to replace it)
numerical_cols = ['bmi']  # 'bmi' is a numerical column we are replacing
train_metadata_remaining_cols = train_metadata.drop(columns=categorical_cols + numerical_cols)
test_metadata_remaining_cols = test_metadata.drop(columns=categorical_cols + numerical_cols)

# Add the processed 'bmi' column back (from imputation)
train_metadata_final = pd.concat([train_metadata_remaining_cols, train_metadata_encoded_df, train_metadata['bmi'].reset_index(drop=True)], axis=1)
test_metadata_final = pd.concat([test_metadata_remaining_cols, test_metadata_encoded_df, test_metadata['bmi'].reset_index(drop=True)], axis=1)

# Optionally, if you want to reset the index to avoid duplicate index values in concatenation
train_metadata_final.reset_index(drop=True, inplace=True)
test_metadata_final.reset_index(drop=True, inplace=True)

# Save the new processed metadata DataFrames
processed_train_metadata_path = os.path.join(project_dir, 'output_resources/processed_train_metadata.csv')
processed_test_metadata_path = os.path.join(project_dir, 'output_resources/processed_test_metadata.csv')

train_metadata_final.to_csv(processed_train_metadata_path, index=False)
test_metadata_final.to_csv(processed_test_metadata_path, index=False)

print(f"Processed train metadata saved to: {processed_train_metadata_path}")
print(f"Processed test metadata saved to: {processed_test_metadata_path}")

#print("Nan values in train data:", pd.isna(train_data).sum().sum()) #Nan values in train data: 1188
#print("Nan values in test data:", pd.isna(test_data).sum().sum())  #0
#print("Nan values in train metadata:", pd.isna(train_metadata_final).sum().sum()) #0
#print("Nan values in test metadata:", pd.isna(test_metadata_final).sum().sum())   #0

#handle nan in train data
numerical_cols = train_data.select_dtypes(include=[np.number]).columns
imputer_num = SimpleImputer(strategy='mean')
train_data[numerical_cols] = imputer_num.fit_transform(train_data[numerical_cols])
test_data[numerical_cols] = imputer_num.transform(test_data[numerical_cols])
#print("Nan values in train data:", pd.isna(train_data).sum().sum()) #0
#print("Nan values in test data:", pd.isna(test_data).sum().sum())  #0

########################################################################
#merge data

train=pd.merge(train_data, train_metadata_final, on='participant_id', how='outer' )
test = pd.merge(test_data, test_metadata_final, on='participant_id', how='outer')
print("The datasets were concatenated with the metadata")
print(train.head())
print(test.head())

#x_train: contains all columns from the training set,
    #excluding the target variable age
#y_train: contains the target value age from the training set
x_train=train.drop(columns=['age'])
y_train=train['age']

x_test=test
x_train_output_path = os.path.join(project_dir, 'output_resources/x_train.csv')
y_train_output_path = os.path.join(project_dir, 'output_resources/y_train.csv')
x_test_output_path = os.path.join(project_dir, 'output_resources/x_test.csv')

x_train.to_csv(x_train_output_path, index=False)  # Save x_train (features) without row index
y_train.to_csv(y_train_output_path, index=False)  # Save y_train (target variable) without row index
x_test.to_csv(x_test_output_path, index=False)    # Save x_test (features) without row index

