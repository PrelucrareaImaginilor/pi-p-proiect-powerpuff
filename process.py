import pandas as pd

#import seaborn as sns

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from Tools.scripts.generate_opcode_h import header

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from scipy.stats import zscore, pearsonr, uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV

#import seaborn as sns

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

def convert_matrix_to_row_vector(matrix):

    # check if it's of type numpy array, then converts to dataframe
    if isinstance(matrix, np.ndarray):
        matrix = pd.DataFrame(matrix)

    #extract the upper triangle
    upper_triangle = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))

    #keep only unique and non-nan values
    unique_correlations = upper_triangle.stack().dropna().unique()

    #transpose the result
    return pd.DataFrame(unique_correlations).T

def generate_column_headers(matrix):
    rows, cols = matrix.shape
    headers = [f'{i}th row_{j}th column' for i in range(rows) for j in range(cols) if i < j]
    return headers

def process_folder(source_folder, output_file):
  vectors = []
  headers=[]
  # iterate over items in the source folder

  for filename in os.listdir(source_folder):
      file_path = os.path.join(source_folder, filename)
      # print(filename)

      # edge cases: check if id is 12 bytes or 11
      len_id = 12 # default
      if filename[4:4+len_id][-1] == "_":
        print(filename[4:4+len_id])
        len_id = 11

      # Check if it is a file (and not a directory) and ends with .tsv
      if os.path.isfile(file_path) and filename.endswith('.tsv'):
          try:
              matrix = pd.read_csv(file_path, sep='\t', header=None)

              # Generate column headers for this matrix
              headers = ['participant_id']+generate_column_headers(matrix)
              all_headers = headers  # Assume all files have the same matrix shape

              # Convert the matrix to a long row of unique correlations
              row_vector = convert_matrix_to_row_vector(matrix)

              # add patient ID
              extracted_id = filename[4:4+len_id]

              row_vector.insert(0, "participant_id", extracted_id)

              # Append the long row as a new row in the results
              vectors.append(row_vector)

          except Exception as e:
              print(f"Error processing file {filename}: {e}")
      else:
          print(f"Skipping {filename} (not a file or not a TSV)")

  # need to write out the file
  out_df = pd.concat(vectors)
  out_df.columns = headers
  out_df.to_csv(output_file, sep='\t', header=True)
  return out_df