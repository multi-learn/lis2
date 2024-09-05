"""
Module with function dealing with missing data
"""
import numpy as np


def test_missing_data_value(image, value):
    """
    Test if there is missing data in the image. Values are
    considered missing if set as NaN or below a given value.

    Parameters
    ----------
    image: np.ndarray
        The input image
    value: float
        The minimal value if not missing

    Returns
    -------
    True if OK, False if some pixel are missing
    """
    res = np.isnan(image)
    if np.any(res):
        return False

    idx = np.where(image < value)
    if idx[0].shape[0] > 0:
        return False

    return True


def fill_missing_data(image, value):
    """
    Fill missing data using the mean of available data (mean imputation)

    Parameters
    ----------
    image: np.ndarray
        The input data
    value: float
        The minimal value if not missing

    Returns
    -------
    The filled data or image
    """
    data = image.flat.copy()
    idx1 = np.nonzero(np.isnan(data))
    idx2 = np.nonzero(data < value)
    idx = np.union1d(idx1, idx2)

    good = np.setdiff1d(np.arange(0, data.shape[0]), idx)
    meanval = data[good].mean()
    data[idx] = meanval

    return data.reshape(image.shape)
