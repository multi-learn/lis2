"""Script to project a set of patches using a classifier."""
import argparse
from deep_filaments.io.utils import kfold_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation from learning method.")
    parser.add_argument("path", help="The path to save the statistics", type=str)
    parser.add_argument("test_dataset_path", help="The path to the hdf5 test dataset", type=str)
    parser.add_argument("train_dataset_path", help="The path to the hdf5 train dataset", type=str)
    parser.add_argument(
        "--batch_size",
        help="The size of the batch",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    kfold_statistics(args.path, args.train_dataset_path, args.test_dataset_path, args.batch_size)