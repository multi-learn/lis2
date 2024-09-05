"""
Normalizing function
"""
import numpy as np
import skimage.exposure as expo


def normalize_direct(x, xmin=None, zmax=None):
    """
    Normalize the data by directly projecting them on [0, 1]

    Parameters
    ----------
    x: np.ndarray
        The data

    Returns
    -------
    A normalized copy of the data
    """
    z = x.copy()
    if xmin is None:
        xmin = np.min(x.flat)
    z -= xmin
    if zmax is None:
        zmax = np.max(z.flat)
    if zmax > 0:
        z /= zmax
    return z


def normalize_histo(x):
    """
    Normalize the data with a histogram equalization

    Parameters
    ----------
    x: np.ndarray
        The data

    Returns
    -------
    A normalized copy of the data
    """
    z = normalize_direct(x)
    z = expo.equalize_hist(z)
    return z


def normalize_adapt_histo(x):
    """
    Normalize the data with a histogram equalization
    TODO: to improve

    Parameters
    ----------
    x: np.ndarray
        The data

    Returns
    -------
    A normalized copy of the data
    """
    z = normalize_direct(x)
    z = np.pad(z, (20, 20), mode="reflect")
    z = expo.equalize_adapthist(z)
    z = z[20 : 20 + x.shape[0], 20 : 20 + x.shape[1]]
    return z
