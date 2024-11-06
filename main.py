from processingTSV import *
import sys


def main():
    """
    train_tsv_path = os.path.join("resources", "train_tsv")
    output_folder = os.path.join("output_vectors")
    process_tsv_files(train_tsv_path, output_folder)
    """

    if len(sys.argv) > 1 and sys.argv[1] == 'process':
        train_tsv_path = os.path.join("resources", "train_tsv")
        output_folder = os.path.join("output_vectors")
        process_tsv_files(train_tsv_path, output_folder)
    else:
        print("No processing was performed. Pass 'process' as an argument.")


if __name__ == '__main__':
    main()