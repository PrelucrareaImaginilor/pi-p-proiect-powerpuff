import numpy as np
import pandas as pd
import os

def extract_upper_triangle(matrix):
    return matrix[np.triu_indices(matrix.shape[0])]

def save_vector_to_file(vector, output_folder, file_name):
    os.makedirs(output_folder, exist_ok=True)
    full_path = os.path.join(output_folder, file_name)
    pd.DataFrame(vector).to_csv(full_path, index=False, header=False)
    print(f"Vector salvat: {full_path}")

def process_matrix_file(input_file, output_folder):
    matrix = pd.read_csv(input_file, sep='\t').values
    print(f"Am incarcat matricea din fisierul: {input_file}")

    upper_triangle = extract_upper_triangle(matrix)
    print(f"Am extras triunghiul superior al matricei.")

    output_file_name = os.path.basename(input_file).replace('.tsv', '_upper_triangle.csv')
    save_vector_to_file(upper_triangle, output_folder, output_file_name)