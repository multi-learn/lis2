"""Functions for direct segmentation of images."""
import numpy as np
import torch


def segment_image(
    image,
    classifier,
    patch_size,
    normalizer=None,
    detect_missing=None,
    fill_missing=None,
):
    """
    Segment image using a projector onto a latent space and a classifier.

    Parameters
    ----------
    image: np.ndarray
        The input image.
    classifier: object
        The classifier (using .predict).
    patch_size: tuple
        The size of one patch.
    normalizer: callable
        A function which normalize input data (None by default).
    detect_missing: callable, optional
        Detect if there is is missing data.
    fill_missing: callable, optional
        Fill detect missing data instead of avoiding the patch (if detect).

    Returns
    -------
    A segmented version of the image.
    """
    res = image.copy() * 0 - 1
    mx = int(patch_size[0] / 2)
    my = int(patch_size[1] / 2)
    offsetx = patch_size[0] % 2
    offsety = patch_size[1] % 2

    if normalizer is not None:
        image = normalizer(image)

    image = np.pad(image, (mx, my), mode="symmetric")

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):

            # 1 - Get the surrounding patch
            patch = image[i : i + 2 * mx + offsetx, j : j + 2 * my + offsety]
            patch = np.reshape(patch, (1, patch_size[0], patch_size[1], 1))

            # 2 - Manage missing data
            if detect_missing is not None and detect_missing(patch):
                if fill_missing is None:
                    continue
                patch = fill_missing(patch)

            # 3 - The classification
            seg_class = None
            if isinstance(classifier, torch.nn.Module):
                # pytorch models
                # should be further tested, dimensions might not be right
                patch = np.reshape(patch, (1, 1, patch_size[0], patch_size[1]))
                patch = patch.astype(np.float32)
                patch = torch.from_numpy(patch)
                with torch.no_grad():
                    seg_class = torch.squeeze(classifier(patch)).numpy()
            else:
                # tf/keras models
                seg_class = classifier.predict(
                    patch.reshape((1, patch_size[0], patch_size[1], 1))
                )

            # 4 - Put
            res[i, j] = seg_class[0, 1]

    return res
