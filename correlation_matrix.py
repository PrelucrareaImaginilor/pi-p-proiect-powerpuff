import os
import pandas as pd
import numpy as np
import glob

def load_vector(patient_id, vector_directory):
    file_pattern = os.path.join(vector_directory, f"sub-{patient_id}_ses-*_correlations_upper_triangle.csv")
    file_path = glob.glob(file_pattern)

    if file_path:
        vector = pd.read_csv(file_path[0], header=None).squeeze()
        print(f"Vector incarcat pentru pacientul {patient_id} din fisierul {file_path[0]}")
        return vector
    else:
        print(f"Vectorul pacientului {patient_id} nu a fost gasit.")
        return None

def create_correlation_matrix(metadata_file, vector_directory):
    metadata = pd.read_csv(metadata_file)
    patient_ids = metadata['participant_id']

    vectors = {}
    for patient_id in patient_ids:
        vector = load_vector(patient_id, vector_directory)
        if vector is not None:
            vectors[patient_id] = vector

    vector_df = pd.DataFrame(vectors)
    correlation_matrix = vector_df.corr()

    return correlation_matrix

if __name__ == "__main__":
    metadata_file = 'resources/metadata/training_metadata.csv'
    vector_directory = 'output_vectors'

    correlation_matrix = create_correlation_matrix(metadata_file, vector_directory)
    print("Matricea de corelare a fost creata.")
    print(correlation_matrix)

    correlation_matrix.to_csv('output_vectors/patient_correlation_matrix.csv')
    print("Matricea de corelare a fost salvata in 'output_vectors/patient_correlation_matrix.csv'")