"""
Script for background maps creation
"""
import argparse
import os

import astropy.io.fits as fits
import numpy as np
import pandas

import deep_filaments.io.utils as utils


def load_threshold_from_file(th_file: str, angles: str, mosaic: str):
    """
    Load the threshold for given angles

    Parameters
    ----------
    th_file: str
        The file with the bounding box list
    angles: str
        The involved angles
    mosaic: str
        The name of the input file

    Returns
    -------
    The threshold
    """
    data = pandas.read_excel(th_file)
    tiles = data["Tile"]
    thresholds = data["Threshold"]
    threshold = 0

    for i in range(tiles.shape[0]):
        if tiles[i] == int(angles):
            threshold = thresholds[i]

    return threshold


def construct_data(mosaic_name: str, threshold: float = 0.0):
    """
    Construct the background from the given data

    Parameters
    ----------
    mosaic_name: str
        The name of the mosaic file
    threshold: float
        The threshold for background segmentation

    Returns
    -------
    The background map
    """
    mosaic_hdu = fits.open(mosaic_name)
    result = np.zeros(mosaic_hdu[0].data.shape)

    nan_idx = np.isnan(mosaic_hdu[0].data)
    idx = mosaic_hdu[0].data < threshold

    result[idx] = 1

    mosaic_hdu[0].data = result
    return mosaic_hdu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Background map construction.")
    parser.add_argument("mosaics", help="The directory with the mosaics", type=str)
    parser.add_argument(
        "threshold", help="File with the threshold for each mosaic", type=str
    )
    parser.add_argument("output", help="The directory for the output files", type=str)
    args = parser.parse_args()

    with os.scandir(args.mosaics) as files:
        for file in files:
            if file.name.endswith(".fits"):
                m_angles = utils.get_mosaic_angles(file.name)
                thresh = load_threshold_from_file(
                    args.threshold, m_angles, args.mosaics + "/" + file.name
                )
                image = construct_data(args.mosaics + "/" + file.name, thresh)
                image_name = "background_mask_" + m_angles + ".fits"
                image.writeto(args.output + "/" + image_name)
