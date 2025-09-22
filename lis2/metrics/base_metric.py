import abc
from typing import Any

import numpy as np
from configurable import TypedConfigurable


class BaseMetric(abc.ABC, TypedConfigurable):
    """
    BaseMetric for defining custom metrics.

    An abstract base class for defining custom metrics. This class provides a structure for implementing
    metrics that can be updated with predictions and targets, and then computed to yield a result.

    Configuration:
        - **name** (str): The name of the metric.

    Example Configuration:
        .. code-block:: python

            config = {
                "name": "example_metric"
            }
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the Metric class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.result = 0.0
        self.averaging_coef = 0.0

    @abc.abstractmethod
    def update(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        idx: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Updates the metric with current predictions and targets.

        Args:
            pred (np.ndarray): Model predictions.
            target (np.ndarray): Ground truth labels.
            idx (np.ndarray): Mask or weighting for valid examples.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def compute(self) -> float:
        """
        Computes the metric result.

        Returns:
            float: The computed metric result.

        Raises:
            ValueError: If no data is available to compute the metric.
        """
        if self.averaging_coef > 0:
            return self.result / self.averaging_coef
        raise ValueError(f"No data to compute metric: {self.name}")

    def reset(self) -> None:
        """
        Resets the metric result and averaging coefficient.
        """
        self.result = 0.0
        self.averaging_coef = 0.0

    def __str__(self) -> str:
        """
        Returns the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self.name
