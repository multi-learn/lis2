from typing import Any

import numpy as np
from configurable import Schema
from skimage.metrics import structural_similarity

from .base_metric import BaseMetric


class MSSIM(BaseMetric):
    """
    MSSIM for evaluating the Mean Structural Similarity Index.

    A metric to compute the Mean Structural Similarity Index (MSSIM) between model predictions and ground truth.
    This metric is useful for evaluating the similarity between two images, particularly in image segmentation tasks.

    Configuration:
        - **name** (str): The name of the metric. Default is "mean_ssim".
        - **threshold** (float): Threshold for binarizing predictions. Default is 0.5.
        - **win_size** (int): Window size for the structural similarity computation. Default is 7.

    Example Configuration:
        .. code-block:: python

            config = {
                "name": "mean_ssim",
                "threshold": 0.5,
                "win_size": 7
            }

    Aliases:
        - `mssim`
        - `ssim`
    """

    config_schema = {
        "win_size": Schema(int, default=7),
        "threshold": Schema(float, default=0.5),
    }
    aliases = ["mssim", "ssim"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the MSSIM metric.

        Args:
            threshold (float): Threshold for binarizing predictions.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "mean_ssim"

    def _core(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Mean Structural Similarity Index (MSSIM).

        Args:
            pred (np.ndarray): Model predictions (flattened numpy array).
            target (np.ndarray): Ground truth (flattened numpy array).

        Returns:
            float: The Mean Structural Similarity Index.
        """
        segmentation = (pred >= self.threshold).astype(np.int32)

        mssim_values = [
            structural_similarity(
                segmentation[i],
                target[i],
                gaussian_weights=False,
                use_sample_covariance=False,
                win_size=self.win_size,
                K1=0.00001,
                K2=0.00001,
                data_range=1,
            )
            for i in range(segmentation.shape[0])
        ]

        return np.mean(mssim_values)

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        idx: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the MSSIM metric with current predictions and targets.

        Args:
            pred (np.ndarray): Model predictions (as probabilities, already flattened numpy arrays).
            target (np.ndarray): Ground truth (already flattened numpy arrays).
            idx (np.ndarray): Mask or weighting for valid examples.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        mssim = self._core(pred, target)
        self.result += mssim * idx.sum().item()
        self.averaging_coef += idx.sum().item()
