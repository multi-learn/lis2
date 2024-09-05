"""Script to project a set of patches using a classifier."""
import argparse

from deep_filaments.utils.informations import read_kfold_statistics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation from learning method.")
    parser.add_argument("path", help="The path to save the statistics", type=str)

    args = parser.parse_args()

    read_kfold_statistics(args.path)