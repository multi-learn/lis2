"""Functions for segmentation of overlapping images."""
import numpy as np
import torch


def generate_patch_set(
    image,
    patch_size,
    overlap=0,
    batch_size=100,
    normalizer=None,
    detect_missing=None,
    missing_map=None,
):
    """
    Create a generation that produces a set of patch of given size from an input image

    Parameters
    ----------
    image: np.ndarray
        The input image
    patch_size: tuple[int, int]
        The size of one patch
    overlap: int, optional
        The overlap between two patches
    batch_size: int
        The maximal size of one batch
    normalizer: callable, optional
        The normalizer.
    detect_missing: callable, optional
        Detect if there is is missing data.
    missing_map: callable, optional
        Compute the missing data map.

    Returns
    -------
    The set of patches with the missing map and the positions
    """
    patches_set = []
    missmap_set = []
    position_set = []
    batch_index = 0

    for i in range(0, image.shape[0], patch_size[0] - overlap):
        find_i_end = False
        if i + patch_size[0] > image.shape[0]:
            find_i_end = True
            i = image.shape[0] - patch_size[0]

        for j in range(0, image.shape[1], patch_size[1] - overlap):
            find_j_end = False
            if j + patch_size[1] > image.shape[1]:
                find_j_end = True
                j = image.shape[1] - patch_size[1]

            # 1 - Get the surrounding patch
            patch = image[i : i + patch_size[0], j : j + patch_size[1]]

            missmap = np.ones(patch.shape)
            if detect_missing is not None and not detect_missing(patch):
                if missing_map is not None:
                    missmap = missing_map(patch)
                else:
                    continue

            if normalizer is not None:
                patch = normalizer(patch)

            patches_set.append(patch)
            missmap_set.append(missmap)
            position_set.append((i, j))
            batch_index += 1

            if batch_index == batch_size:
                yield np.array(patches_set), np.array(missmap_set), position_set
                batch_index = 0
                patches_set.clear()
                missmap_set.clear()
                position_set.clear()

            if find_j_end:
                break

        if find_i_end:
            break

    # Produce the rest if needed
    if batch_index > 0:
        yield np.array(patches_set), np.array(missmap_set), position_set


def segment_with_generator(patchesgen, image, segmenter=None):
    """
    Recreate an image from a given set of patches

    Parameters
    ----------
    patchesgen: Generator
        A generator set of patches
    image: np.ndarray
        The reference image (TODO: use only the size)
    segmenter: callable
        The segmenter (should be able to take a batch of patches)

    Returns
    -------
    The rebuild image from the patches
    """
    res = np.zeros(image.shape)
    covering = np.zeros(image.shape)
    size = 0

    for patches, missmaps, positions in patchesgen:

        p_i = patches.shape[1]
        p_j = patches.shape[2]
        size += len(positions)

        # Do segmentation
        seg = patches
        if segmenter is not None:
            patches = np.reshape(patches, (patches.shape[0], 1, p_i, p_j))
            patches = patches.astype(np.float32)
            patches = torch.from_numpy(patches)
            with torch.no_grad():
                seg = torch.squeeze(segmenter(patches)).numpy()

        seg *= missmaps
        for pos in range(len(positions)):
            i, j = positions[pos][0], positions[pos][1]
            res[i : i + p_i, j : j + p_j] += seg[pos, :, :]
            covering[i : i + p_i, j : j + p_j] += np.ones((p_i, p_j))

        print("Number of processed patches = {}".format(size))

    # Overlap correction
    covering[covering == 0] = 1
    res /= covering

    return res


# overlap related to scanning / stride
def overlap_segmentation(
    image,
    segmenter,
    patch_size,
    overlap=0,
    normalizer=None,
    detect_missing=None,
    missing_map=None,
    use_segmenter=True,
):
    """
    Segmentation with overlapping images.

    Parameters
    ----------
    image: np.ndarray
        The input image (full size).
    segmenter: callable
        The predictor for segmentation.
    patch_size: tuple
        The size of one patch.
    overlap: int, optional
        The number of overlapping pixels between 2 patches.
    normalizer: callable, optional
        The normalizer.
    detect_missing: callable, optional
        Detect if there is is missing data.
    missing_map: callable, optional
        Compute the missing data map.
    use_segmenter: boolean, optional
        If True we use the segmenter, else only the identity operator is applied
    Returns
    -------
    The segmented image.
    """
    res = np.zeros(image.shape)
    covering = np.zeros(res.shape)

    for i in range(0, res.shape[0], patch_size[0] - overlap):
        find_i_end = False
        if i + patch_size[0] > res.shape[0]:
            find_i_end = True
            i = res.shape[0] - patch_size[0]

        for j in range(0, res.shape[1], patch_size[1] - overlap):
            find_j_end = False
            if j + patch_size[1] > res.shape[1]:
                find_j_end = True
                j = res.shape[1] - patch_size[1]

            # 1 - Get the surrounding patch
            patch = image[i : i + patch_size[0], j : j + patch_size[1]]

            idx = np.isnan(patch)
            patch[idx] = 0
            missmap = np.ones(patch.shape)
            if detect_missing is not None and not detect_missing(patch):
                if missing_map is not None:
                    missmap = missing_map(patch)
                else:
                    continue

            if normalizer is not None:
                midx = missmap > 0
                if midx.any():
                    patch[midx] = normalizer(patch[midx])

            # 3 - The segmentation
            patch = np.reshape(patch, (1, 1, patch_size[0], patch_size[1]))
            if use_segmenter:
                patch = patch.astype(np.float32)
                patch = torch.from_numpy(patch)
                with torch.no_grad():
                    seg = torch.squeeze(segmenter(patch)).numpy()
            else:
                seg = np.squeeze(patch)

            # 4 - Put and get the overlap
            res[i : i + patch_size[0], j : j + patch_size[1]] += missmap * seg
            covering[i : i + patch_size[0], j : j + patch_size[1]] += np.ones(seg.shape)

            if find_j_end:
                break

        if find_i_end:
            break

    # 5 - Overlap correction
    covering[covering == 0] = 1
    res /= covering

    return res
