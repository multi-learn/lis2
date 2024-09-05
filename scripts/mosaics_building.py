"""
Building mosaics using several tiles/images
"""
import argparse

import astropy.io.fits as fits
import numpy as np
import reproject
import reproject.mosaicking

import deep_filaments.io.utils as utils


def build_mosaic(files_dir):
    """
    For each file get the data from the other using merging

    Parameters
    ----------
    files_dir: str
        The directory with the files for the composition

    Returns
    -------
    The blended results for each input file with the corresponding filenames
    """
    files = utils.get_sorted_file_list(files_dir)
    hdus = [fits.open(files_dir + "/" + f) for f in files]

    new_hdus = []
    for hdu in hdus:
        header = hdu[0].header.copy()

        a, f = reproject.mosaicking.reproject_and_coadd(
            hdus, header, reproject_function=reproject.reproject_interp
        )
        idx = a > 0
        a[idx] = 1

        new_hdu = fits.ImageHDU(data=a, header=header)
        new_hdus.append(new_hdu)

    return new_hdus, files


def build_unified_mosaic(
    files_dir,
    naxis1,
    naxis2,
    hdu_number=0,
    avoid_missing=False,
    missing_value=1.0,
    binarize=False,
    conservative=False,
):
    """
    Build a mosaic using all the files inside a directory

    Parameters
    ----------
    files_dir: str
        The directory with the files for the composition
    naxis1: int
        The width of the new image
    naxis2: int
        The height of the new image
    hdu_number: int, optional
        The number of the HDU inside the fits
    avoid_missing: bool, optional
        True if we want to avoid missing values (below 1) problems
    missing_value: float, optional
        The threshold for detecting missing values
    binarize: bool, optional
        True for a binarized result
    conservative: bool, optional
        If True, apply a conservative binarization

    Returns
    -------
    The a full mosaic file
    """
    files = utils.get_sorted_file_list(files_dir)
    hdus = [(fits.open(files_dir + "/" + f))[hdu_number] for f in files]
    new_header = hdus[0].header.copy()
    new_header["NAXIS1"] = naxis1
    new_header["NAXIS2"] = naxis2
    new_header["CRPIX1"] = naxis1 // 2
    new_header["CRPIX2"] = naxis2 // 2
    new_header["CRVAL1"] = 180.0
    new_header["CRVAL2"] = 0.0

    # Convert all missing values into NaN (avoid stupid means)
    if avoid_missing:
        for hdu in hdus:
            idx = hdu.data < missing_value
            hdu.data[idx] = np.nan

    a, f = reproject.mosaicking.reproject_and_coadd(
        hdus, new_header, reproject_function=reproject.reproject_interp
    )

    # Put everything to 0 if not filament
    if binarize:
        a[np.isnan(a)] = 0.0
        if conservative:
            a[a < 0.6] = 0.0
            a[a > 0] = 1.0
        else:
            a[a > 0.2] = 1.0
            a[a < 0.5] = 0.0

    return a, new_header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mosaic of images.")
    parser.add_argument("files_dir", help="The directory with the files", type=str)
    parser.add_argument("output_dir", help="The output_directory", type=str)
    parser.add_argument(
        "--one_file",
        help="Merging in one single file",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hdu_number",
        help="The number of the involved HDU inside the files",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--avoid_missing",
        help="Manage missing value below the missing value",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--missing_value",
        help="The threshold value for missing data",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--binarize",
        help="Binarize the result",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--conservative",
        help="Apply a conservative binarization",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if not args.one_file:
        reshdus, files_list = build_mosaic(args.files_dir)

        for fhdu, file in zip(reshdus, files_list):
            fhdu.writeto(args.output_dir + "/" + file)
    else:
        data, header = build_unified_mosaic(
            args.files_dir,
            114000,
            1800,
            args.hdu_number,
            args.avoid_missing,
            args.missing_value,
            args.binarize,
            args.conservative,
        )
        fits.writeto(args.output_dir + "/merge_result.fits", data=data, header=header, overwrite=True)
