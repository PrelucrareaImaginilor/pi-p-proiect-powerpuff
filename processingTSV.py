import os
from Laura import *
from George import *

def process_all_tsv_files(input_directory, output_directory):
    tsv_files = laura.list_tsv_files(input_directory)

    if not tsv_files:
        print(f"Nu am gasit fisiere TSV in {input_directory}.")
        return

    for tsv_file in tsv_files:
        print(f"Procesam fisierul: {tsv_file}")

        data = laura.load_tsv(tsv_file)

        if data.empty:
            print(f"Salt fisierul {tsv_file} pentru ca este gol.")
            continue

        george.process_matrix_file(tsv_file, output_directory)

    print("Procesarea fisierelor TSV a fost completata.")

if __name__ == "__main__":
    input_directory = 'resources/train_tsv'
    output_directory = 'output_vectors'

    process_all_tsv_files(input_directory, output_directory)
