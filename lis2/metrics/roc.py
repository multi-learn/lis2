from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .base_metric import BaseMetric


class ROCAUCScore(BaseMetric):
    """
    ROCAUCScore for evaluating the area under the ROC curve.

    A metric to compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) score, which is a measure
    of the ability of a classifier to distinguish between classes. This metric is particularly useful for binary
    classification problems.

    Configuration:
        - **name** (str): The name of the metric. Default is "roc_auc".

    Example Configuration:
        .. code-block:: python

            config = {
                "name": "roc_auc"
            }

    Aliases:
        - `roc_auc`
        - `auc`
    """

    aliases = ["roc_auc", "auc"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the ROCAUCScore metric.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "roc_auc"

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        idx: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the ROCAUCScore metric with current predictions and targets.

        Args:
            pred (np.ndarray): Model predictions.
            target (np.ndarray): Ground truth labels.
            idx (np.ndarray): Mask or weighting for valid examples.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pred = np.round(pred).astype(int).flatten()
        target = np.round(target).astype(int).flatten()
        idx = idx.flatten()
        roc_auc = roc_auc_score(target, pred)
        self.result += roc_auc * idx.sum().item()
        self.averaging_coef += idx.sum().item()
