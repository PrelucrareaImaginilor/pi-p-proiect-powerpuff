import os
import torch
from torch.utils.data import Dataset, DataLoader
from Laura import *
from George import *
import pandas as pd

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")
#print(f"Using {device} device")

class TSVFileDataset(Dataset):
    def __init__(self, input_folder):
        self.file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tsv')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        df = load_tsv(file_path)

        if df is not None:
            matrix = df.values
            vector = extract_upper_triangular(matrix)
            vector = torch.tensor(vector, dtype=torch.float32).to(device)
            return vector, file_path
        return None, None

def process_tsv_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    dataset = TSVFileDataset(input_folder) #an instance of TSVFileDataset class
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=10, pin_memory=True)

    for batch, file_paths in dataloader:
        for vector, file_path in zip(batch, file_paths):
            if vector is not None:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file_name = f"vector_{base_name}.tsv"
                output_path = os.path.join(output_folder, output_file_name)

                # save vector to output file
                vector_cpu = vector.cpu().numpy()  # Move vector back to CPU for saving
                save_vectors_to_tsv_as_rows(vector_cpu, output_path)
                print(f"Saved vector to {output_file_name}")

