import numpy as np
import pandas as pd

def extract_upper_triangular(matrix):
    matrix = np.array(matrix)
    upper_triangular_indices = np.triu_indices_from(matrix, k=1)
    return matrix[upper_triangular_indices]

def save_vectors_to_csv(vectors, file_path):
    df = pd.DataFrame(vectors)
    df.to_csv(file_path, index=False)

def save_vectors_to_npy(vectors, file_path):
    np.save(file_path, np.array(vectors))