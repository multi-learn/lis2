"""
Compare two segmentations results for filaments
"""
import argparse
import os
import astropy.io.fits as fits
import deep_filaments.io.utils as utils
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure the quality of a segmentation."
    )
    parser.add_argument("segmentation", help="The segmentation result", type=str)
    parser.add_argument("image", help="The segmentation result", type=str)
    parser.add_argument("normed_image", help="The segmentation result", type=str)
    parser.add_argument("spines", help="The groundtruth", type=str)
    parser.add_argument("background", help="The background", type=str)
    parser.add_argument("--folder", help="The output database file", type=str, default="results"
    )
    parser.add_argument(
        "--model",
        help="The network model",
        default="UNet",
        type=str,
        choices=["UNet", "UNetPP", "SwinUNet", "DnCNN"],
    )
    parser.add_argument(
        "--file_prefix",
        help="Set the file profile (default: unet_segmentation)",
        type=str,
        default="unet_segmentation",
    )
    args = parser.parse_args()

    if args.image.endswith(".fits"):
        img = fits.getdata(args.image)
        idx = np.isnan(img)
        img[idx] = 0
        header = fits.getheader(args.image)

    if args.normed_image.endswith(".fits"):
        normed_image = fits.getdata(args.normed_image)
        idx = np.isnan(normed_image)
        normed_image[idx] = 0
    
    if args.segmentation.endswith(".fits"):
        segmentation_map = fits.getdata(args.segmentation)
        idx = np.isnan(segmentation_map)
        segmentation_map[idx] = 0

    if args.spines.endswith(".fits"):
        spines = fits.getdata(args.spines)
        idx = np.isnan(spines)
        spines[idx] = 0

    if args.background.endswith(".fits"):
        background = fits.getdata(args.background)
        idx = np.isnan(background)
        background[idx] = 0

    local_segmentation_binarize_map, global_segmentation_binarize_map = utils.segmentation_performances(segmentation_map, normed_image, spines, background, os.path.join(args.folder, args.file_prefix + "_segmenation_performances.hdf5"), args.model)
    utils.save_segmentation(local_segmentation_binarize_map, global_segmentation_binarize_map, img, spines, header, args.folder, args.file_prefix)