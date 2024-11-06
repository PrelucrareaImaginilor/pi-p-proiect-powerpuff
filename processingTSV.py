import os
from Laura import *
from George import *

def process_tsv_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tsv_files = [f for f in os.listdir(input_folder) if f.endswith('.tsv')]

    for file_name in tsv_files:
        file_path = os.path.join(input_folder, file_name)
        data_frame = load_tsv(file_path)
        if data_frame is not None:
            matrix = data_frame.values
            vector = extract_upper_triangular(matrix)
            output_file_name = f"vector_{file_name[:-4]}.tsv"
            output_path = os.path.join(output_folder, output_file_name)

            pd.DataFrame(vector).to_csv(output_path, sep='\t', index=False, header=False)
            print(f"Vector extracted and saved from {file_name} to {output_path}")

