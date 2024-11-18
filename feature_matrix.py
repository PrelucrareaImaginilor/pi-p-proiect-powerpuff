import os
import pandas as pd
import glob

from sympy.logic.inference import valid


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

def create_feature_matrix(metadata_file, vector_directory):
    metadata = pd.read_csv(metadata_file)
    patient_ids = metadata['participant_id']

    feature_data = []

    for patient_id in patient_ids:
        vector = load_vector(patient_id, vector_directory)
        if vector is not None:
            patient_metadata = metadata[metadata['participant_id'] == patient_id].iloc[0].to_dict()
            vector_columns = {f'vector_element_{i}': val for i, val in enumerate(vector)}
            patient_metadata.update(vector_columns)
            feature_data.append(patient_metadata)

    feature_matrix = pd.DataFrame(feature_data)
    return feature_matrix



if __name__ == "__main__":
    metadata_file = 'resources/metadata/training_metadata.csv'
    vector_directory = 'output_vectors'

    feature_matrix = create_feature_matrix(metadata_file, vector_directory)
    print("Matricea de feature a fost creata.")
    "print(correlation_matrix)"

    feature_matrix.to_csv('output_vectors/patient_feature_matrix.csv', index=False)
    print("Matricea de feature a fost salvata in 'output_vectors/patient_feature_matrix.csv'")