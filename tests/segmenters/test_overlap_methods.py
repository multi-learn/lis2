from unittest import TestCase

import numpy as np
import torch

import deep_filaments.segmenters.overlap_methods as overlap


class Segmenter:
    def __call__(self, x):
        """
        Stupid segement
        Parameters
        ----------
        x: np.ndarray
            Input array

        Returns
        -------
        1 if the sum of array element is above 1, 0 else.
        """
        if x.sum() > 1:
            return torch.zeros(x.shape)
        return torch.ones(x.shape)


def normalizer(x):
    """
    Simple normalizer
    Parameters
    ----------
    x: np.ndarray
        Input array

    Returns
    -------
    The normalized input (summing at 1)
    """
    z = x / x.sum()
    return z


class Test(TestCase):
    def test_overlap_segmentation(self):
        img = np.ones((10, 10))
        seg1 = overlap.overlap_segmentation(img, Segmenter(), (2, 2))
        self.assertTrue((seg1 == np.zeros((10, 10))).all())
        seg2 = overlap.overlap_segmentation(img, Segmenter(), (2, 2), 0, normalizer)
        self.assertTrue((seg2 == np.ones((10, 10))).all())

    def test_segment_with_generator(self):
        img = np.ones((10, 10))
        gen = overlap.generate_patch_set(img, (2, 2), 1)
        seg1 = overlap.segment_with_generator(gen, img, Segmenter())
        self.assertTrue((seg1 == np.zeros((10, 10))).all())
