from processingTSV import *
import sys
from Laura import *
from processingVectors import *

def main():

    train_tsv_path = os.path.join("resources", "train_tsv")
    output_folder = os.path.join("output_vectors")

    if os.path.exists(output_folder) and os.listdir(output_folder):
        print ("Output vectors are ready.")
    else:
        print("Output vectors are processing...")
        process_tsv_files(train_tsv_path, output_folder)

    check_vector_dimensions(output_folder, num_vectors=3)
    # size of a row vector: [1, 19900], which is correct bc we had 200x200 matrices
    # the output vectors should have a dimension of 200*199/2

    metadata_file = "resources/metadata/test_metadata.csv"
    full_feature_tensor = create_feature_matrix_with_metadata(output_folder, metadata_file)
    print("Feature Matrix created\n")
    output_feature_matrix = os.path.join("output_feature_matrix")
    save_feature_matrix_with_metadata(full_feature_tensor, output_feature_matrix)
    print("Feature Matrix saved\n")

if __name__ == '__main__':
    main()