from processingTSV import *
import sys


def main():

    #if len(sys.argv) > 1 and sys.argv[1] == 'process':
    train_tsv_path = os.path.join("resources", "train_tsv")
    output_folder = os.path.join("output_vectors")
    process_tsv_files(train_tsv_path, output_folder)
    print("Done")
    #else:
        #print("No processing was performed. Pass 'process' as an argument.")


if __name__ == '__main__':
    main()