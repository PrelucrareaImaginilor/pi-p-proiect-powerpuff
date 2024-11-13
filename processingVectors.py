from Laura import *
from George import *
import os
import torch

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_output_vectors(output_folder):
    vectors = []
    participant_ids = []

    for file in os.listdir(output_folder):
        if file.endswith('.tsv'):
            file_path=os.path.join(output_folder, file)
            vector = pd.read_csv(file_path, sep='\t', header=None)
            participant_id = extract_participant_id(file)

            vectors.append(vector)
            participant_ids.append(participant_id)

    feature_matrix = np.vstack(vectors)

    return feature_matrix, participant_ids

def load_metadata(metadata_file, participant_ids):
    # Load metadata and set the index to participant_id for easy alignment
    metadata_df = pd.read_csv(metadata_file)

    metadata_df.set_index('participant_id', inplace=True)

    metadata_df.infer_objects(copy=False)

    # Reindex to match participant_ids, filling any missing metadata with NaN
    #metadata_aligned = metadata_df.reindex(participant_ids).fillna(0) # Fill NaN if necessary
    metadata_aligned = metadata_df.reindex(participant_ids)  # Fill NaN if necessary

    metadata_values = pd.to_numeric(metadata_aligned.values, errors='coerce')
    metadata_values = np.nan_to_num(metadata_values, nan=0.0)
    # Optionally, convert metadata to a PyTorch tensor and move to device (CPU/GPU)
    metadata_tensor = torch.tensor(metadata_aligned.values, dtype=torch.float32).to(device)
    return metadata_tensor

def create_feature_matrix_with_metadata(output_folder, metadata_file):
    """Combine vectors and metadata into a single feature matrix based on participant_id alignment."""
    # Load vectors and associated participant IDs
    feature_matrix, participant_ids = load_output_vectors(output_folder)

    # Convert feature matrix to PyTorch tensor and move to device
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)

    # Load and align metadata to match the order of participant IDs
    metadata_tensor = load_metadata(metadata_file, participant_ids)

    # Concatenate feature tensor and metadata tensor along columns
    full_feature_tensor = torch.cat([feature_tensor, metadata_tensor], dim=1)

    return full_feature_tensor


def save_feature_matrix_with_metadata(full_feature_tensor, output_feature_matrix_filename):
    """Save the full feature matrix with metadata to a TSV file."""
    # Move tensor to CPU and convert to NumPy for saving
    output_feature_matrix=os.path.abspath(output_feature_matrix_filename)
    if not os.path.exists(output_feature_matrix):
        with open(output_feature_matrix, 'w'): pass  # Create an empty file

    full_feature_matrix = full_feature_tensor.cpu().numpy()

    output_df = pd.DataFrame(full_feature_matrix)
    output_df.to_csv(output_feature_matrix, sep='\t', index=False, header=False)
    print(f"Full feature matrix saved to {output_feature_matrix}")