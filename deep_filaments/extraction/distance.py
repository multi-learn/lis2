"""
Distance function important for skeleton building

Implementation from "A GENERAL ALGORITHM FOR COMPUTING DISTANCE TRANSFORMS IN LINEAR TIME" by
A. MEIJSTER‚ J.B.T.M. ROERDINK and W.H. HESSELINK
"""
import numpy as np


def euclidean_function(x, i, g):
    """
    Euclidean distance function

    Parameters
    ----------
    x: float
        The x position
    i: float
        The i position
    g: float
        The distance between i and another point (given from formula)

    Returns
    -------
    The distance between x and i
    """
    return (x - i) ** 2 + g**2


def _first_scanning(img):
    """
    First scanning function for distance computation

    Parameters
    ----------
    img: np.ndarray
        The image binary image

    Returns
    -------
    The intermediate distance map
    """

    g = img * 0.0
    for i in range(img.shape[0]):
        if img[i, 0] > 0:
            g[i, 0] = np.inf

        for j in range(1, img.shape[1]):
            if img[i, j] > 0:
                g[i, j] = 1 + g[i, j - 1]
            else:
                g[i, j] = 0

        for j in range(img.shape[1] - 2, -1, -1):
            if g[i, j + 1] < g[i, j]:
                g[i, j] = 1 + g[i, j + 1]

    return g


def _sep(i, u, x, y):
    """
    Separation distance

    Parameters
    ----------
    i: float
        The i position
    u: float
        The u position
    x: float
        g(u) = G(u, y) (with y from paper)
    y: float
        g(i) = G(i, y) (with y from paper)

    Returns
    -------
    The separation distance between i and u
    """
    if u != i:
        temp = u**2 - i**2 + x**2 - y**2
        res = temp // (2 * (u - i))
        return res

    return 0.0


def _second_scanning(img, g, f):
    """
    Second scanning function for computing the distance map

    Parameters
    ----------
    img: np.ndarray
        The binary shape image
    g: np.ndarray
        The intermediate distance map
    f: callable
        The distance function

    Returns
    -------
    The distance map
    """
    distance = img * 0.0
    s = np.zeros((np.max([img.shape[0], img.shape[1]]),), dtype="i")
    t = np.zeros((np.max([img.shape[0], img.shape[1]]),), dtype="i")
    for j in range(img.shape[1]):
        q = 0
        s[0] = 0
        t[0] = 0
        for i in range(1, img.shape[0]):
            while q >= 0 and f(t[q], s[q], g[s[q], j]) > f(t[q], i, g[i, j]):
                q -= 1

            if q < 0:
                q = 0
                s[0] = i
            else:
                w = 1 + _sep(s[q], i, g[i, j], g[s[q], j])
                if w < img.shape[0]:
                    q += 1
                    s[q] = i
                    t[q] = w

        for i in range(img.shape[0] - 1, -1, -1):
            distance[i, j] = f(i, s[q], g[s[q], j])
            if i == t[q]:
                q -= 1

    return distance


def distance_map(img):
    """
    Compute the distance map of one binary shape

    Parameters
    ----------
    img: np.ndarray
        A binary image with a shape

    Returns
    -------
    The distance map (distance to the border)
    """
    g = _first_scanning(img)
    distance = _second_scanning(img, g, euclidean_function)
    return np.sqrt(distance)
