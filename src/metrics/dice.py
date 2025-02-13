from typing import Any

import numpy as np
from configurable import Schema

from src.metrics.average_precision import BaseMetric


class Dice(BaseMetric):
    config_schema = {"threshold": Schema(float, default=0.5)}
    aliases = ["dice", "dice_index"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the Dice metric.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "dice"

    def update(self, pred: np.ndarray, target: np.ndarray, idx: np.ndarray, **kwargs: Any) -> None:
        """
        Updates the Dice metric with current predictions and targets.

        Args:
            pred (np.ndarray): Model predictions.
            target (np.ndarray): Ground truth labels.
            idx (np.ndarray): Mask or weighting for valid examples.
            **kwargs: Arbitrary keyword arguments.
        """
        segmentation = (pred >= self.threshold).astype(int)
        self.result += self._core(segmentation, target) * idx.sum().item()
        self.averaging_coef += idx.sum().item()

    def _core(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the Dice coefficient.

        Args:
            pred (np.ndarray): Model predictions (binary mask).
            target (np.ndarray): Ground truth labels (binary mask).

        Returns:
            float: The Dice coefficient.
        """
        if pred.max().item() > 1:
            raise ValueError("The segmented_images tensor should be a 0-1 map.")

        if target.max().item() > 1:
            raise ValueError("The groundtruth_images tensor should be a 0-1 map.")

        seg_data_tp = pred + target
        tp_value = 2
        tp = (seg_data_tp == tp_value).sum().item()
        segData_FP = 2 * pred + target
        segData_FN = pred + 2 * target
        fp = (segData_FP == 2).sum().item()
        fn = (segData_FN == 2).sum().item()
        if 2 * tp + fp + fn > 0:
            return 2 * tp / (2 * tp + fp + fn)
        return 1.0
