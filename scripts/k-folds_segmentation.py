"""Script to project a set of patches using a classifier."""
import argparse
import sys

from deep_filaments.torch.models import UNet, UNetPP, SwinUNet, DnCNN, UNet2, UNet2_pe, UNet_pe
import deep_filaments.io.utils as utils

import torch
import numpy as np
import astropy.io.fits as fits
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation from learning method.")
    parser.add_argument("weights", help="The NN model file (HDF5)", type=str)
    parser.add_argument("input", help="The input file", type=str)
    parser.add_argument("source", help="The source file .fits", type=str)
    parser.add_argument("-o", "--output", help="The output file", type=str, default="")
    parser.add_argument(
        "--normalization_mode",
        help="Normalize patches before segmentation, should be the same than during training",
        type=str,
        default="none",
        choices=["direct", "log10", "none"],
    )
    parser.add_argument(
        "--no_segmenter",
        help="Do not apply the neural network to segment the input",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model",
        help="The network model",
        default="UNet",
        type=str,
        choices=["UNet", "UNetPP", "SwinUNet", "DnCNN", "UNet2", "UNet_pe", "UNet2_pe"],
    )
    parser.add_argument(
        "--batch_size", help="The size of the batch (new way)", default=100, type=int
    )
    parser.add_argument(
        "--missing",
        help="Do not apply the neural network to segment the input",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    source_data = fits.getdata(args.source)
    source_header = fits.getheader(args.source)
    segmentation_map = np.zeros_like(source_data)
    count_map = segmentation_map.copy()
    
    dataset_files = utils.get_sorted_file_list(args.input)
    model_files = utils.get_sorted_file_list(args.weights)

    for i in range(len(dataset_files)):
        if model_files[i].endswith(".pt"):
            model = None
            if args.model == "UNetPP":
                model = UNetPP()
            elif args.model == "SwinUNet":
                model = SwinUNet()
            elif args.model == "DnCNN":
                model = DnCNN()
            elif args.model == "UNet":
                model = UNet()
            elif args.model == "UNet2":
                model = UNet2()
            elif args.model == "UNet_pe":
                model = UNet_pe()
            elif args.model == "UNet2_pe":
                model = UNet2_pe()
            else:
                sys.exit(1)
            model.load_state_dict(
                torch.load(
                    os.path.join(args.weights, model_files[i]),
                )
            )
            model.eval()
            model.to(device)
        else:
            sys.exit(1)

        segmentation_map, count_map = utils.segmentation(os.path.join(args.input, dataset_files[i]), model, segmentation_map, count_map, args.normalization_mode, args.missing, args.batch_size, args.no_segmenter, device)
    idx = count_map > 0
    segmentation_map[idx] = segmentation_map[idx] / count_map[idx]
    fits.writeto(args.output, data=segmentation_map, header=source_header, overwrite=True)