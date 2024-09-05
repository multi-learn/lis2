"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""
import argparse

import numpy as np
import astropy.io.fits as fits
from deep_filaments.io.utils import extract_patches

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch dataset construction.")
    parser.add_argument("image", help="The n(h2) image", type=str)
    parser.add_argument("--roi", help="The corresponding RoI", type=str, default=None)
    parser.add_argument("--missing", help="The missing data mask", type=str, default=None)
    parser.add_argument("--background", help="The background data", type=str, default=None)
    parser.add_argument(
        "-o", "--output", help="The output database file", type=str, default="seg"
    )
    parser.add_argument(
        "--train_overlap", help="The overlap between two patches of the training set", type=int, default=0
    )
    parser.add_argument(
        "--test_overlap", help="The overlap between two patches of the testing set", type=int, default=0
    )
    parser.add_argument(
        "--mask", help="The mask", type=str, default=None
    )
    parser.add_argument(
        "--patch_size",
        help="The size of one patch (default: 64x64)",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--train",
        help="The dataset created will be compatible with the training procedure if this parameter is set to True. If the parameter 'mask' is given, this parameter won't be taken into account",
        action="store_true",
        default=False,
    )
    
    args = parser.parse_args()

    # Manage default patch size
    if args.patch_size is None:
        patches_size = (64, 64)
    else:
        patches_size = tuple(args.patch_size)

    if args.image.endswith(".fits"):
        img = fits.getdata(args.image)
        idx = np.where(np.isnan(img))
        img[idx] = 0

    if args.roi is None:
        roi = None
    elif args.roi.endswith(".fits"):
        roi = fits.getdata(args.roi)

    if args.missing is None:
        missing = None
    elif args.missing.endswith(".fits"):
        missing = fits.getdata(args.missing)

    if args.background is None:
        background = None
    elif args.background.endswith(".fits"):
        background = fits.getdata(args.background)

    if args.mask is None:
        mask = None
    elif args.mask.endswith(".npy"):
        mask = np.load(args.mask)
    elif args.mask.endswith(".fits"):
        mask = fits.getdata(args.mask)
    else:
        print('--mask argument has to ends with ".npy"')

    hdfs = extract_patches(
        img,
        args.output,
        patches_size,
        train=args.train,
        target=roi,
        missing=missing,
        background=background,
        train_overlap=args.train_overlap,
        test_overlap=args.test_overlap,
        mask=mask
    )
    [hdf.close() for hdf in hdfs]
