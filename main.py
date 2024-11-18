from feature_matrix import create_feature_matrix
from Laura import *


def main():

    input_directory = 'resources/train_tsv'
    output_directory = 'output_vectors'
    metadata_file = 'resources/metadata/training_metadata.csv'

    # print("Starting TSV processing pipeline...")
    # process_all_tsv_files(input_directory, output_directory)
    # print("TSV processing pipeline completed successfully.")

    print("Start! Matricea de feature incepe...")
    feature_matrix = create_feature_matrix(metadata_file, output_directory)
    print("Matrice de feature creata..")

    output_file = 'output_vectors/patient_correlation_matrix.csv'
    feature_matrix.to_csv(output_file, index=False)
    print(f"Matricea de corelare a fost salvata la: '{output_file}'.")


if __name__ == "__main__":
    main()
