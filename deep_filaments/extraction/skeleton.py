"""
Skeleton construction algorithms see paper
 [SBTZ02] K. Siddiqi, S. Bouix, A. Tannenbaum and S.W. Zucker. Hamilton-Jacobi Skeletons
    International Journal of Computer Vision, 48(3):215-231, 2002
"""
import heapq

import numpy as np
import scipy.interpolate as sci
import skimage
import numba
from numba.typed import List


@numba.jit
def _check_limits(x, i, j):
    """
    Test if (i, j) is correct coordinates for x

    Parameters
    ----------
    x: np.ndarray
        Input array
    i: int
        first coordinate
    j: int
        second coordinate

    Returns
    -------
    True if OK, False else
    """
    if i < 0 or j < 0 or i >= x.shape[0] or j >= x.shape[1]:
        return False
    return True


@numba.jit
def _is_end_point(image, pos):
    """
    Test if a given pixel is an end-point

    Parameters
    ----------
    image: np.ndarray
        The current shape
    pos: list[int]
        The position of the pixel

    Returns
    -------
    True if end-point, False else
    """
    # 1 - Get the number of neighbors
    count = 0
    for k in [-1, 0, 1]:
        for l in [-1, 0, 1]:
            if k == 0 and l == 0:
                continue
            if image[pos[0] + k, pos[1] + l] > 0:
                count += 1

    if count == 1:
        return True

    # If 2, we have 8 possible configuration to test
    if count == 2:
        if image[pos[0] + 1, pos[1]] > 0 and image[pos[0] + 1, pos[1] + 1] > 0:
            return True
        if image[pos[0] + 1, pos[1]] > 0 and image[pos[0] + 1, pos[1] - 1] > 0:
            return True
        if image[pos[0] - 1, pos[1]] > 0 and image[pos[0] - 1, pos[1] + 1] > 0:
            return True
        if image[pos[0] - 1, pos[1]] > 0 and image[pos[0] - 1, pos[1] - 1] > 0:
            return True
        if image[pos[0], pos[1] + 1] > 0 and image[pos[0] + 1, pos[1] + 1] > 0:
            return True
        if image[pos[0], pos[1] + 1] > 0 and image[pos[0] - 1, pos[1] + 1] > 0:
            return True
        if image[pos[0], pos[1] - 1] > 0 and image[pos[0] + 1, pos[1] - 1] > 0:
            return True
        if image[pos[0], pos[1] - 1] > 0 and image[pos[0] - 1, pos[1] - 1] > 0:
            return True

    return False


@numba.jit
def _is_simple_point(image, pos):
    """
    Test if a given pixel is simple. We assume the point is not at a border (pad if needed)

    **Warning**: this code only works for 2D and 8-connexity

    Parameters
    ----------
    image: np.ndarray
        The input image
    pos: List[int]
        The coordinate of the point

    Returns
    -------
    True if simple, False else
    """
    # 1 - Get the number of nodes
    nodes = 0
    for k in [-1, 0, 1]:
        for l in [-1, 0, 1]:
            if k == 0 and l == 0:
                continue
            if image[pos[0] + k, pos[1] + l] > 0:
                nodes += 1

    # 2 - Get the number of edges
    edges = 0

    # Test direct relation
    if image[pos[0] - 1, pos[1] - 1] > 0 and image[pos[0] - 1, pos[1]] > 0:
        edges += 1
    if image[pos[0] - 1, pos[1] - 1] > 0 and image[pos[0], pos[1] - 1] > 0:
        edges += 1
    if image[pos[0] - 1, pos[1] + 1] > 0 and image[pos[0] - 1, pos[1]] > 0:
        edges += 1
    if image[pos[0] - 1, pos[1] + 1] > 0 and image[pos[0], pos[1] + 1] > 0:
        edges += 1
    if image[pos[0] + 1, pos[1] - 1] > 0 and image[pos[0] + 1, pos[1]] > 0:
        edges += 1
    if image[pos[0] + 1, pos[1] - 1] > 0 and image[pos[0], pos[1] - 1] > 0:
        edges += 1
    if image[pos[0] + 1, pos[1] + 1] > 0 and image[pos[0] + 1, pos[1]] > 0:
        edges += 1
    if image[pos[0] + 1, pos[1] + 1] > 0 and image[pos[0], pos[1] + 1] > 0:
        edges += 1
    # Test diagonal relation (avoid cycle)
    if (
        image[pos[0] - 1, pos[1]] > 0
        and image[pos[0], pos[1] - 1] > 0
        and image[pos[0] - 1, pos[1] - 1] < 1
    ):
        edges += 1
    if (
        image[pos[0] - 1, pos[1]] > 0
        and image[pos[0], pos[1] + 1] > 0
        and image[pos[0] - 1, pos[1] + 1] < 1
    ):
        edges += 1
    if (
        image[pos[0] + 1, pos[1]] > 0
        and image[pos[0], pos[1] - 1] > 0
        and image[pos[0] + 1, pos[1] - 1] < 1
    ):
        edges += 1
    if (
        image[pos[0] + 1, pos[1]] > 0
        and image[pos[0], pos[1] + 1] > 0
        and image[pos[0] + 1, pos[1] + 1] < 1
    ):
        edges += 1

    if nodes - edges == 1:
        return True
    return False


@numba.jit(forceobj=True)
def compute_flux(image, s_x=1.0, s_y=1.0):
    """
    Compute the flux of the gradient

    Parameters
    ----------
    image: np.ndarray
        The input "distance" image
    s_x: float, optional
        The x-size of one pixel
    s_y: float, optional
        The y-size of one pixel

    Returns
    -------
    The flux of the image
    """
    flux = image * 0.0

    img_gradient_x = skimage.filters.prewitt_h(image)
    interp_grad_x = sci.interp2d(
        range(image.shape[0]),
        range(image.shape[1]),
        img_gradient_x.T,
        kind="cubic",
        copy=False,
    )

    img_gradient_y = skimage.filters.prewitt_v(image)
    interp_grad_y = sci.interp2d(
        range(image.shape[0]),
        range(image.shape[1]),
        img_gradient_y.T,
        kind="cubic",
        copy=False,
    )

    for i_x in range(image.shape[0]):
        for i_y in range(image.shape[1]):

            f = 0.0
            if image[i_x, i_y] < 1e-9:
                continue

            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (k == 0 and l == 0) or not _check_limits(
                        image, i_x + k, i_y + l
                    ):
                        continue
                    norm = np.sqrt((s_x * k) ** 2 + (s_y * l) ** 2)
                    f += (
                        interp_grad_x(i_x + k * s_x / norm, i_y + l * s_y / norm) * k
                        + interp_grad_y(i_x + k * s_x / norm, i_y + l * s_y / norm) * l
                    ) / norm

            flux[i_x, i_y] = f / 8.0

    # flux /= 2.0 * np.pi
    return flux


@numba.jit(forceobj=True)
def compute_skeleton(image, distances, threshold):
    """
    Compute the medial axis skeleton

    Parameters
    ----------
    image: np.ndarray
        The binary input image
    distances: np.ndarray
        The distance image
    threshold: float
        The threshold on the flux (robustness to noise)

    Returns
    -------
    The skeleton of the input image
    """
    skeleton = image.copy()
    count = image * 0

    # 0 - Compute flux from distances
    flux = compute_flux(distances)

    # 1 - Init with the bound points
    queue = []
    for i_x in range(image.shape[0]):
        for i_y in range(image.shape[1]):

            if image[i_x, i_y] < 1e-9:
                continue

            border = False
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (k == 0 and l == 0) or not _check_limits(
                        image, i_x + k, i_y + l
                    ):
                        continue
                    if image[i_x + k, i_y + l] < 1e-9:
                        border = True
                        break
                if border:
                    break

            if border and _is_simple_point(image, List((i_x, i_y))):
                heapq.heappush(
                    queue, (-flux[i_x, i_y], distances[i_x, i_y], List((i_x, i_y)))
                )
                count[i_x, i_y] = 1

    # 2 - Skeletonization loop
    while len(queue) > 0:
        element = heapq.heappop(queue)
        count[element[2][0], element[2][1]] = 1  # Removing the count marker
        if _is_simple_point(skeleton, List(element[2])):
            if (
                skeleton[element[2][0], element[2][1]] != 2
                and not _is_end_point(skeleton, List(element[2]))
            ) or -element[0] > threshold:
                skeleton[element[2][0], element[2][1]] = 0

                for k in [-1, 0, 1]:
                    for l in [-1, 0, 1]:
                        if (k == 0 and l == 0) or count[
                            element[2][0] + k, element[2][1] + l
                        ] == 1:
                            continue

                        if (
                            not _check_limits(
                                skeleton, element[2][0] + k, element[2][1] + l
                            )
                            or skeleton[element[2][0] + k, element[2][1] + l] < 1e-9
                        ):
                            continue

                        if _is_simple_point(
                            skeleton, List((element[2][0] + k, element[2][1] + l))
                        ):
                            heapq.heappush(
                                queue,
                                (
                                    -flux[element[2][0] + k, element[2][1] + l],
                                    distances[element[2][0] + k, element[2][1] + l],
                                    (element[2][0] + k, element[2][1] + l),
                                ),
                            )
                            count[element[2][0] + k, element[2][1] + l] = 1
            else:
                skeleton[element[2][0], element[2][1]] = 2

    return skeleton
