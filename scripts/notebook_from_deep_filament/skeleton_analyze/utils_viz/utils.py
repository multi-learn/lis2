import pandas as pd

import numpy as np

import skimage.morphology as morph
import skimage.measure as measure
import skimage.io as io

def filter_around_zero(data, threshold):
    return data[(data < -threshold) | (data > threshold)]


def get_skeleton(mask2D, option="opencv") :
    skeleton = morph.skeletonize(mask2D)

    return skeleton


def get_skeleton_instances(skeleton):
    labeled_skeleton, num_labels = measure.label(skeleton, return_num=True, connectivity=2)
    skeleton_images = []
    for label in range(1, num_labels + 1):
        # Créer une image pour le squelette courant
        individual_skeleton = (labeled_skeleton == label).astype(np.uint8)
        skeleton_images.append(individual_skeleton)
    return skeleton_images


def get_histogram(smoothed_signal, common_bin_edges):

    smoothed_signal_hist = filter_around_zero(smoothed_signal, threshold=0.2)
    data_hist, bin_edges_Y = np.histogram(smoothed_signal_hist, bins=common_bin_edges, density=True)
    bin_centers = 0.5 * (common_bin_edges[:-1] + common_bin_edges[1:])

    return data_hist, bin_centers