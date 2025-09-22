from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from .base_metric import BaseMetric


class AveragePrecision(BaseMetric):
    """
    AveragePrecision for evaluating the average precision score.

    A metric to compute the average precision score, which summarizes the precision-recall curve as the area under the curve.
    This metric is particularly useful for evaluating the performance of binary classifiers. It is based on the
    `average_precision_score` function from `sklearn.metrics`.

    Configuration:
        - **name** (str): The name of the metric. Default is "average_precision".

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "average_precision"

    Aliases:
        - `average_precision`
        - `ap`
        - `map`
    """

    aliases = ["average_precision", "ap", "map"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the AveragePrecision metric.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "average_precision"

    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        idx: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the Average Precision metric with current predictions and targets.

        Args:
            pred (np.ndarray): Model predictions, rounded for binary masks.
            target (np.ndarray): Ground truth labels (binary).
            idx (np.ndarray): Mask or weighting for valid examples.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pred = np.round(pred).astype(int)
        target = np.round(target).astype(int)

        ap = average_precision_score(target.flatten(), pred.flatten())
        self.result += ap * idx.sum().item()
        self.averaging_coef += idx.sum().item()
