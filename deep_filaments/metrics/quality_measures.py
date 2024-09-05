"""
Function for qualitative comparison
"""
import numpy as np
import skimage.morphology as morpho


def positive_part_accuracy(segmented_images, groundtruth_images):
    """
    Calculate the accuracy of the positive part (the class 1).

    Parameters
    ----------
    segmented_images: np.ndarray
        Segmentation results as 0-1 map.
    groundtruth_images: np.ndarray
        Target segmentation image as 0-1 map.

    Returns
    -------
    The accuracy of the segmentation compare to the known positive part
    """
    res = (segmented_images * groundtruth_images).sum() / groundtruth_images.sum()
    return res


def get_missed_structures(image, gtruth, threshold):
    """
    Get the missed structures in a segmentation

    Parameters
    ----------
    image: np.ndarray
        The new segmentation file
    gtruth: np.ndarray
        The ground truth
    threshold: float
        The threshold for segmentation

    Returns
    -------
    The map of missed structures
    """
    seg = np.zeros(image.shape)
    seg[image >= threshold] = 1.0

    res = morpho.reconstruction(seg * gtruth, gtruth)

    res[res > 0] = 1
    miss_labels = gtruth - res

    return miss_labels


def compute_missed_structures(image, gtruth, threshold):
    """
    Compute the percentage of missed structures in a segmentation

    Parameters
    ----------
    image: np.ndarray
        The new segmentation file
    gtruth: np.ndarray
        The ground truth
    threshold: float
        The threshold for segmentation

    Returns
    -------
    The percentage of missed structures
    """
    seg = np.zeros(image.shape)
    seg[image >= threshold] = 1.0

    res = morpho.reconstruction(seg * gtruth, gtruth)

    res[res > 0] = 1
    miss_labels = morpho.label(gtruth - res, connectivity=2)
    true_labels = morpho.label(gtruth, connectivity=2)

    nb_miss_labels = np.max(miss_labels)
    nb_true_labels = np.max(true_labels)

    if np.isnan(nb_true_labels) or nb_true_labels < 1:
        nb_true_labels = 1

    return nb_miss_labels / nb_true_labels


def compare_segmentation(image, gtruth, threshold):
    """
    Compare a segmentation result with groundtruth using a given threshold

    Parameters
    ----------
    image: np.ndarray
        The segmentation result (probability image)
    gtruth: np.ndarray
        The groundtruth
    threshold: float
        The threshold (must be positive)

    Returns
    -------
    The percentage of recovered structures in the image
    """
    seg = np.zeros(image.shape)
    seg[image >= threshold] = 1.0
    intersection = seg * gtruth

    return intersection.sum() / gtruth.sum()
