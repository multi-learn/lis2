from unittest import TestCase

import numpy as np

import deep_filaments.segmenters.direct_methods as direct


class Classifier:
    @staticmethod
    def predict(x):
        """
        Stupid classifier
        Parameters
        ----------
        x: np.ndarray
            Input array

        Returns
        -------
        1 if the sum of array element is above 1, 0 else.
        """
        if x.sum() > 1:
            return np.array([[0, 1]])
        return np.array([[1, 0]])


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
    def test_segment_image(self):
        img = np.ones((10, 10))
        seg1 = direct.segment_image(img, Classifier, (2, 2))
        self.assertTrue((seg1 == np.ones((10, 10))).all())
        seg2 = direct.segment_image(img, Classifier, (2, 2), normalizer)
        self.assertFalse((seg2 == np.ones((10, 10))).all())
