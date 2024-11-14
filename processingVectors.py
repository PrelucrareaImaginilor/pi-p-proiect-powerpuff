from Laura import *
from George import *
import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureMatrixDataset(Dataset):
    def __init__(self, input_folder, metadata_file):
        self.metadata_file = metadata_file
        self.file_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tsv')]
        self.participant_ids = [extract_participant_id(f) for f in os.listdir(input_folder) if f.endswith('.tsv')]

        # Load and align metadata
        self.metadata_tensor, self.metadata_columns = self._load_and_align_metadata()

        # Load and align metadata
        self.metadata_df, self.metadata_columns = self._load_and_align_metadata(metadata_file)

    def _load_and_align_metadata(self, metadata_file):

        metadata_df = pd.read_csv(self.metadata_file)
        metadata_df.set_index('participant_id', inplace=True)

        # Reindex to match the order of participant IDs
        metadata_aligned = metadata_df.reindex(self.participant_ids)

        # Convert to a numeric tensor, filling NaN with 0 if necessary
        metadata_values = metadata_aligned.apply(pd.to_numeric, errors='coerce').fillna(0).values

        '''
        metadata_columns = metadata_aligned.columns.tolist()  # Get the column names of the metadata
        # Return both tensor and columns
        return torch.tensor(metadata_values, dtype=torch.float32).to(device), metadata_columns
    '''
        metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32).to(device)
        return metadata_tensor, metadata_aligned.columns.tolist()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # Load the TSV vector for the specified participant
        file_path = self.file_paths[index]
        vector = pd.read_csv(file_path, sep='\t', header=None).values.flatten()  # Flatten to a 1D array

        # Convert vector to a PyTorch tensor and move to the specified device
        vector_tensor = torch.tensor(vector, dtype=torch.float32).to(device)

        # Retrieve the aligned metadata for this participant
        metadata_tensor = self.metadata_df[index]

        # Concatenate vector and metadata along columns
        full_feature = torch.cat([vector_tensor, metadata_tensor], dim=0)

        return full_feature, self.participant_ids[index]


def create_feature_matrix_with_metadata(output_folder, metadata_file):
    dataset = FeatureMatrixDataset(output_folder, metadata_file)
    dataloader = DataLoader(dataset, batch_size=len(dataset),
                            shuffle=False)  # Load all at once for a single feature matrix

    # Retrieve the feature matrix and participant IDs
    for full_feature_tensor, participant_ids in dataloader:
        # Return only the tensor, not the tuple (which includes participant_ids)
        return full_feature_tensor.to(device), participant_ids, dataset.metadata_columns  # Add metadata column names


def save_feature_matrix_with_metadata(full_feature_tensor, participant_ids, metadata_columns,
                                      output_feature_matrix_filename):

    # Convert tensor to NumPy array
    full_feature_matrix = full_feature_tensor.cpu().numpy()

    # Prepare a list of column names: first the vector features, then the metadata columns
    num_features = full_feature_matrix.shape[1] - len(metadata_columns)  # Number of vector features
    feature_columns = [str(i) for i in range(num_features)]  # Vector columns: 0, 1, 2, ..., N-1
    full_columns = feature_columns + metadata_columns  # Combined columns (vector + metadata)

    # Create DataFrame and save as TSV
    output_df = pd.DataFrame(full_feature_matrix, columns=full_columns)
    output_df.insert(0, 'participant_id', participant_ids)
    output_df.to_csv(output_feature_matrix_filename, sep='\t', index=False)
    print(f"Full feature matrix saved to {output_feature_matrix_filename}")
