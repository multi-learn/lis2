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
    parser.add_argument("normed_image", help="The n(h2) normed image", type=str)
    parser.add_argument("roi", help="The corresponding RoI", type=str)
    parser.add_argument("missing", help="The missing data mask", type=str)
    parser.add_argument("background", help="The background data", type=str)
    parser.add_argument(
        "-o", "--output", help="The output database file", type=str, default="dataset"
    )
    parser.add_argument(
        "--train_overlap", help="The overlap between two patches of the training set", type=int, default=0
    )
    parser.add_argument(
        "--kfold_overlap", help="The overlap between two area of the k-fold", type=int, default=0
    )
    parser.add_argument(
        "--test_overlap", help="The overlap between two patches of the testing set", type=int, default=0
    )
    parser.add_argument(
        "--test_area_size", help="Size of k-fold regions", type=int, default=62
    )
    parser.add_argument(
        "--patch_size",
        help="The size of one patch (default: 64x64)",
        nargs="+",
        type=int,
    )
    parser.add_argument(
        "--k",
        help="Number of fold",
        type=int,
        default=10
    )
    parser.add_argument(
        "--k_fold_mode",
        help="Mode of operation the k-fold split",
        type=str,
        default="naive",
        choices=["naive", "random"],
    )
    
    args = parser.parse_args()

    # Manage default patch size
    if args.patch_size is None:
        patches_size = (64, 64)
    else:
        patches_size = tuple(args.patch_size)

    if args.image.endswith(".fits"):
        img = fits.getdata(args.image)

    if args.normed_image.endswith(".fits"):
        normed_image = fits.getdata(args.normed_image)

    if args.roi.endswith(".fits"):
        roi = fits.getdata(args.roi)

    if args.missing.endswith(".fits"):
        missing = fits.getdata(args.missing)

    if args.background.endswith(".fits"):
        background = fits.getdata(args.background)

    test_area_size = args.test_area_size
    test_fold_mask = np.array([np.zeros_like(img) for _ in range(args.k)])
    if args.k_fold_mode == "random":
        x, y = 0, 0
        while y + test_area_size <= img.shape[1]:
            fold = np.random.randint(0, args.k)
            x = 0
            while x + test_area_size <= img.shape[0]:
                test_fold_mask[fold][x : x + test_area_size, y : y + test_area_size] = 1
                fold = (fold + 1) % args.k
                x += test_area_size - args.kfold_overlap
                if img.shape[0] - test_area_size < x < img.shape[0] - args.kfold_overlap:
                    x = img.shape[0] - test_area_size
            y += test_area_size - args.kfold_overlap
            if img.shape[1] - test_area_size < y < img.shape[1] - args.kfold_overlap:
                y = img.shape[1] - test_area_size
    else:
        for i in range(len(test_fold_mask)):
            test_fold_mask[i][:, int(i * img.shape[1] / args.k) : int((i + 1) * img.shape[1] / args.k)] = 1

    hdfs_fold = [
        extract_patches(
        img,
        args.output + f"fold_{i}",
        patches_size,
        normed_image=normed_image,
        target=roi,
        missing=missing,
        background=background,
        train_overlap=args.train_overlap,
        test_overlap=args.test_overlap,
        mask=test_fold_mask[i]
    )
    for i in range(len(test_fold_mask))
    ]
    for hdfs in hdfs_fold:
        [hdf.close() for hdf in hdfs]