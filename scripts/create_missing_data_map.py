"""
Script creating missing data map from a set a tiles
"""
import argparse
import os

import astropy.io.fits as fits
import astropy.wcs as wcs
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
import pandas

import deep_filaments.io.utils as utils


def load_bb_from_file(bb_file: str, angles: str, mosaic: str):
    """
    Load the bounding boxes for given angles

    Parameters
    ----------
    bb_file: str
        The file with the bounding box list
    angles: str
        The involved angles
    mosaic: str
        The name of the input file

    Returns
    -------
    A list with the bounding boxes, None if empty
    """
    data = pandas.read_excel(bb_file)
    tiles = data["Tile"]
    up_bb_l = data["Up BB l"]
    up_bb_b = data["Up BB b"]
    down_bb_l = data["Down BB l"]
    down_bb_b = data["Down BB b"]
    res = []

    mosaic_hdu = fits.open(mosaic)
    coord_sys = wcs.WCS(mosaic_hdu[0].header)

    for i in range(tiles.shape[0]):
        if tiles[i] == int(angles):
            up_c = coord.SkyCoord(
                coord.Galactic(l=up_bb_l[i] * u.deg, b=up_bb_b[i] * u.deg)
            )
            down_c = coord.SkyCoord(
                coord.Galactic(l=down_bb_l[i] * u.deg, b=down_bb_b[i] * u.deg)
            )
            bb_coord = [
                coord_sys.world_to_pixel(up_c),
                coord_sys.world_to_pixel(down_c),
            ]
            res.append(bb_coord)

    if res:
        return res
    return None


def construct_data(mosaic_name: str, threshold: float = 0.0, bb_list: list = None):
    """
    Construction of the data map using filaments and mosaics

    Parameters
    ----------
    mosaic_name: str
        The name of the mosaic file
    threshold: float
        The threshold value (below it is a missing value)
    bb_list: list, optional
        A list of bounding boxes to include in missing data

    Returns
    -------
    The map of all the ROI on the mosaic
    """
    mosaic_hdu = fits.open(mosaic_name)
    result = np.ones(mosaic_hdu[0].data.shape)

    idx = np.isnan(mosaic_hdu[0].data)
    result[idx] = 0

    idx = mosaic_hdu[0].data < threshold
    result[idx] = 0

    if bb_list:
        for bb in bb_list:
            x_t, y_t = int(bb[0][0]), int(bb[0][1])
            x_b, y_b = int(bb[1][0]), int(bb[1][1])

            if x_t < 0:
                x_t = 0
            if y_b < 0:
                y_b = 0
            if y_t > result.shape[0]:
                y_t = result.shape[0]
            if x_b > result.shape[1]:
                x_b = result.shape[1]

            result[y_b:y_t, x_t:x_b] = 0

    mosaic_hdu[0].data = result
    return mosaic_hdu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Missing data map construction.")
    parser.add_argument("mosaics", help="The directory with the mosaics", type=str)
    parser.add_argument("output", help="The directory for the output files", type=str)
    parser.add_argument(
        "--filter",
        help="Filter part of the tile using BB from a given file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--missing_value",
        help="The threshold value for missing data",
        type=float,
        default=0.1,
    )
    args = parser.parse_args()

    with os.scandir(args.mosaics) as files:
        for file in files:
            if file.name.endswith(".fits"):
                m_angles = utils.get_mosaic_angles(file.name)
                l_bb = None
                if args.filter:
                    l_bb = load_bb_from_file(
                        args.filter, m_angles, args.mosaics + "/" + file.name
                    )

                image = construct_data(
                    args.mosaics + "/" + file.name, args.missing_value, l_bb
                )
                image_name = "missing_data_mask_" + m_angles + ".fits"
                image.writeto(args.output + "/" + image_name)
