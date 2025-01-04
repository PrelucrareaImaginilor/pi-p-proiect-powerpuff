import os
from process import process_folder
import pandas as pd

source_folder_train = 'resources/train_tsv'
source_folder_test = 'resources/test_tsv'

output_file_train = 'output_resources/train_output.tsv'
output_file_test = 'output_resources/test_output.tsv'

project_dir = os.path.dirname(__file__)

source_folder_train_path = os.path.join(project_dir, source_folder_train)
source_folder_test_path = os.path.join(project_dir, source_folder_test)

output_file_train_path = os.path.join(project_dir, output_file_train)
output_file_test_path = os.path.join(project_dir, output_file_test)

if not os.path.exists(source_folder_train_path):
    raise FileNotFoundError(f"Source folder does not exist: {source_folder_train_path}")

if not os.path.exists(source_folder_test_path):
    raise FileNotFoundError(f"Source folder does not exist: {source_folder_test_path}")

output_dir = os.path.join(project_dir, 'output_resources')
os.makedirs(output_dir, exist_ok=True)

train_data=pd.DataFrame()
test_data=pd.DataFrame()

if not os.path.exists(output_file_train_path):
    train_data = process_folder(source_folder_train_path, output_file_train_path)
    print(f"Train data processed and saved to: {output_file_train_path}")
if not os.path.exists(output_file_test_path):
    test_data = process_folder(source_folder_test_path, output_file_test_path)
    print(f"Test data processed and saved to: {output_file_test_path}")

metadata_train = 'resources/metadata/training_metadata.csv'
metadata_test = 'resources/metadata/test_metadata.csv'

metadata_train_path = os.path.join(project_dir, metadata_train)
metadata_test_path = os.path.join(project_dir, metadata_test)

if not os.path.exists(metadata_train_path):
    raise FileNotFoundError(f"Metadata file does not exist: {metadata_train_path}")

if not os.path.exists(metadata_test_path):
    raise FileNotFoundError(f"Metadata file does not exist: {metadata_test_path}")

#################################################
if os.path.exists(output_file_train_path) and os.path.exists(output_file_test_path):
    train_data = pd.read_csv(output_file_train_path, sep='\t')
    test_data = pd.read_csv(output_file_test_path, sep='\t')
    print("Loaded test and train data")

#print("Nan values in train data:", pd.isna(test_data).sum().sum())
#print("Nan values in test data:", pd.isna(train_data).sum().sum())

train_metadata=pd.read_csv(metadata_train_path)
print(train_metadata.head())
test_metadata=pd.read_csv(metadata_test_path)
print(test_metadata.head())

'''
train=pd.merge(train_data, train_metadata, on='participant_id', how='outer' )
test = pd.merge(test_data, test_metadata, on='participant_id', how='outer')
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
'''