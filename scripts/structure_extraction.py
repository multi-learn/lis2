"""Script to extract structures inside a segmentation map"""
import argparse

import numpy as np
import skimage.morphology as skm
import skimage.segmentation as sks
import skimage.filters as skf
import matplotlib.pyplot as plt
import astropy.io.fits as fits

import deep_filaments.extraction.distance as e_dst
import deep_filaments.extraction.skeleton as e_ske
import deep_filaments.extraction.structures as e_str


def compute_structures(
    image,
    density,
    threshold,
    erosion_size,
    skel_threshold,
    use_density=False,
    distances_sigma=0.5,
):
    """Compute structures inside an image

    Parameters
    ----------
    image: np.ndarray
        The input image
    density: np.ndarray
        The N(H2) density map
    threshold: float
        The binarization threshold
    erosion_size: int
        The size of the square for erosion (0 => no erosion)
    skel_threshold: float
        The threshold for skeleton extraction
    use_density: bool
        If True use N(H2) density as distances
    distances_sigma: float
        The sigma for smooth the distances using a Gaussian filter

    Returns
    -------
    The skeleton of the structures inside the input image and the roi (region of interest)
    """
    # 0 - Binarization
    image[image > threshold] = 1
    image[image < 1] = 0

    # 1 - Removing small structures
    if erosion_size > 0:
        erode = skm.erosion(image, skm.square(erosion_size))
        image = skm.reconstruction(erode, image)

    # 2 - Skeleton (to improve)
    if not use_density:
        distances = e_dst.distance_map(image)
        distances = skf.gaussian(distances, sigma=distances_sigma)
    else:
        distances = density
    skel = e_ske.compute_skeleton(image, distances, skel_threshold)
    skel[skel > 0] = 1  # Remove end-line pixels

    return skel, image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction of structures inside a given image."
    )
    parser.add_argument("input", help="The input file", type=str)
    parser.add_argument("density", help="The density map", type=str)
    parser.add_argument(
        "-t", "--threshold", help="The threshold for detection", type=float, default=0.8
    )
    parser.add_argument(
        "-s",
        "--skel_threshold",
        help="The threshold for extraction",
        type=float,
        default=-0.3,
    )
    parser.add_argument(
        "-d",
        "--display",
        help="Display the results",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-o", "--output", help="Output file (FITS format)", type=str, default=None
    )
    parser.add_argument(
        "-e", "--erode", help="Erode before computing skeleton", type=int, default=0
    )
    args = parser.parse_args()

    hdu = fits.open(args.input)
    img = np.array(hdu[0].data, dtype="f4")

    hdu2 = fits.open(args.density)
    img_density = hdu2[0].data

    skeleton, roi = compute_structures(
        img, img_density, args.threshold, args.erode, args.skel_threshold
    )
    if args.display:
        plt.imshow(skeleton, origin="lower")
        plt.show()

    labels = skm.label(skeleton, connectivity=2)
    roi_labels = sks.watershed(
        roi, markers=labels, mask=roi, connectivity=np.ones((3, 3))
    )

    print("Numbers of structures = {}".format(labels.max()))

    if args.display:
        plt.imshow(labels, origin="lower")
        plt.show()
        plt.imshow(roi_labels * roi, origin="lower")
        plt.show()

    if args.output:
        # TODO: use density map for statistics
        structures = e_str.structures_extraction(labels, roi_labels, img_density)
        structures.to_excel(args.output + ".xlsx")

        hdu[0].data = np.array(labels, dtype=np.float64)
        hdu.writeto(args.output + "-labels.fits")

        hdu[0].data = np.array(roi_labels, dtype=np.float64)
        hdu.writeto(args.output + "-roi.fits")
