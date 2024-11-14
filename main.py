from processingTSV import process_all_tsv_files
from correlation_matrix import create_correlation_matrix


def main():

    input_directory = 'resources/train_tsv'
    output_directory = 'output_vectors'
    metadata_file = 'resources/metadata/training_metadata.csv'

    # print("Starting TSV processing pipeline...")
    # process_all_tsv_files(input_directory, output_directory)
    # print("TSV processing pipeline completed successfully.")

    print("Start! Matricea de corelare incepe...")
    correlation_matrix = create_correlation_matrix(metadata_file, output_directory)
    print("Matrice de corelare creata..")

    output_file = 'output_vectors/patient_correlation_matrix.csv'
    correlation_matrix.to_csv(output_file)
    print(f"Matricea de corelare a fost salvata la: '{output_file}'.")


if __name__ == "__main__":
    main()
