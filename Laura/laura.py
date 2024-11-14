import pandas as pd
import os

def load_tsv(file_path):
    try:
        data = pd.read_csv(file_path, sep='\t')
        print(f"Incarcat informatii din {file_path}")
        return data
    except Exception as e:
        print(f"Eroare la incarcare fisier {file_path}: {e}")
        return pd.DataFrame()

def list_tsv_files(directory):
    tsv_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tsv')]
    print(f"Am gasit {len(tsv_files)} fisiere TSV in {directory}")
    return tsv_files