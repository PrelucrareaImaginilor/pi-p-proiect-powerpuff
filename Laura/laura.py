import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import re

def load_tsv(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t', header=None)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: There was a parsing error while reading the file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def save_vectors_to_tsv_as_rows(vectors, file_path):
    #df=pd.DataFrame(vectors, columns=[None])
    df=pd.DataFrame([vectors])
    df.to_csv(file_path,sep='\t', index=False, header=False)

def check_vector_dimensions(output_folder, num_vectors=2):
    output_files = [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.tsv')]

    for output_file in output_files[:num_vectors]:
        vector_df = pd.read_csv(output_file, sep='\t', header=None)
        vector = torch.tensor(vector_df.values, dtype=torch.float32)
        print(f"Vector shape: {vector.shape}")

#we get the participant_id for each subject's vector
# we are looking for ex: NDARAT358XM9
def extract_participant_id(filename):
    match= re.search(r'NDAR[A-Z]{2}\d{3}[A-Z0-9]{3}', filename)
    if match:
        return match.group()
    return None

