import pandas as pd

def load_tsv(file_path):
    try:
        df = pd.read_csv(file_path, sep='\t')
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