import os
from Laura import *
from George import *

def process_tsv_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tsv_files = [f for f in os.listdir(input_folder) if f.endswith('.tsv')]

    for file_name in tsv_files:
        file_path = os.path.join(input_folder, file_name)
        df = load_tsv(file_path)

        if df is not None:
            matrix = df.values

            vector = extract_upper_triangular(matrix)
            base_name = os.path.splitext(os.path.basename(input_folder))[0]
            output_file_name = f"vector_{base_name}.tsv"
            output_path = os.path.join(output_folder, output_file_name)

            save_vectors_to_tsv_as_rows(vector, output_path)
            #print(f"Vector extracted and saved from {file_name} to {output_path}")





