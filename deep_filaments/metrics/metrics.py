"""Metrics for evaluation of algorithm performance."""
import torch


def positive_part_accuracy(segmented_images, groundtruth_images):
    """
    Calculate the accuracy of the positive part (the class 1).

    Parameters
    ----------
    segmented_images: torch.Tensor
        Segmentation results as 0-1 map.
    groundtruth_images : torch.Tensor
        Target segmentation image as 0-1 map.

    Returns
    -------
    The accuracy of the segmentation compare to the known positive part
    """
    nb = groundtruth_images.sum().item()
    res = 1.0
    if nb > 0:
        res = (segmented_images * groundtruth_images).sum().item() / nb
    return res


def dice_index(segmented_images, groundtruth_images, missing_data=None):
    """
    Calculate the DICE similarity score.

    Parameters
    ----------
    segmented_images : torch.Tensor
        Segmentation results as 0-1 map.
    groundtruth_images : torch.Tensor
        Target segmentation image as 0-1 map.
    missing_data: torch.Tensor
        Missing data map

    Returns
    -------
    float
        DICE similarity score.
    """
    if missing_data is not None:
        idx = missing_data > 0
        segmented_images = segmented_images[idx]
        groundtruth_images = groundtruth_images[idx]

    if segmented_images.max().item() > 1:
        raise ValueError("The segmented_images tensor should be a 0-1 map.")
    if groundtruth_images.max().item() > 1:
        raise ValueError("The groundtruth_images tensor should be a 0-1 map.")

    segData_TP = segmented_images + groundtruth_images
    TP_value = 2
    # true positive: segmentation result and groundtruth match(both are positive)
    TP = (segData_TP == TP_value).sum().item()
    segData_FP = 2 * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2 * groundtruth_images
    # false positive: segmentation result and groundtruth mismatch
    FP = (segData_FP == 2).sum().item()
    # false negative: segmentation result and groundtruth mismatch
    FN = (segData_FN == 2).sum().item()
    # according to the definition of the DICE similarity score
    if 2 * TP + FP + FN > 0:
        return 2 * TP / (2 * TP + FP + FN)
    return 1.0

def average_precision(segmented_images, groundtruth_images, missing_data=None):
    """
    Calculate the average precision score given by the product of precision and recall.

    Parameters
    ----------
    segmented_images : torch.Tensor
        Segmentation results as 0-1 map.
    groundtruth_images : torch.Tensor
        Target segmentation image as 0-1 map.
    missing_data: torch.Tensor
        Missing data map

    Returns
    -------
    float
        average precision score.
    """
    if missing_data is not None:
        idx = missing_data > 0
        segmented_images = segmented_images[idx]
        groundtruth_images = groundtruth_images[idx]

    if segmented_images.max().item() > 1:
        raise ValueError("The segmented_images tensor should be a 0-1 map.")
    if groundtruth_images.max().item() > 1:
        raise ValueError("The groundtruth_images tensor should be a 0-1 map.")

    segData_TP = segmented_images + groundtruth_images
    TP_value = 2
    # true positive: segmentation result and groundtruth match(both are positive)
    TP = (segData_TP == TP_value).sum().item()
    segData_FP = 2 * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2 * groundtruth_images
    # false positive: segmentation result and groundtruth mismatch
    FP = (segData_FP == 2).sum().item()
    # false negative: segmentation result and groundtruth mismatch
    FN = (segData_FN == 2).sum().item()
    if TP + FN > 0 and TP + FP > 0:
        recall =  TP / (TP + FN)
        precision = TP / (TP + FP)
        return recall * precision
    return 0.0


def pixel_acc(segmented_images, groundtruth_images, missing_data=None):
    """
    Calculate the pixel accuracy score.

    Parameters
    ----------
    segmented_images : torch.Tensor
        Segmentation results as 0-1 map.
    groundtruth_images : torch.Tensor
        Target segmentation image as 0-1 map.
    missing_data: torch.Tensor
        Missing data map

    Returns
    -------
    float
        Pixel accuracy.
    """
    if missing_data is not None:
        idx = missing_data > 0
        segmented_images = segmented_images[idx]
        groundtruth_images = groundtruth_images[idx]

    if segmented_images.max().item() > 1:
        raise ValueError("The segmented_images tensor should be a 0-1 map.")
    if groundtruth_images.max().item() > 1:
        raise ValueError("The groundtruth_images tensor should be a 0-1 map.")
    if segmented_images.shape != groundtruth_images.shape:
        raise ValueError("Images should be of the same shape")

    correct = torch.sum(segmented_images == groundtruth_images)
    total = segmented_images.numel()

    if total > 0:
        return correct.item() / total
    return 1.0
