"""
Script for displaying the loss/dice values from a training experiments
"""

import argparse
from deep_filaments.utils.informations import read_segmentation_perf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display learning curves from experiments file (HDF5)."
    )
    parser.add_argument(
        "input_file", help="The experiments file (in hdf5 format)", type=str
    )
    args = parser.parse_args()
    read_segmentation_perf(args.input_file)