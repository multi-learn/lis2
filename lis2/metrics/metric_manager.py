from typing import List, Dict, Any

from tqdm import tqdm

from .base_metric import BaseMetric


class MetricManager:
    def __init__(self, metrics_configs: List[Dict[str, Any]]):
        """
        Initializes the Metrics class with a list of metric configurations.

        Args:
            metrics_configs (List[Dict[str, Any]]): List of metric configurations.
        """
        self.metrics = [BaseMetric.from_config(config) for config in metrics_configs]

    def update(self, *args: Any, **kwargs: Any) -> None:
        """
        Updates all metrics with the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        errors = []
        for metric in tqdm(self.metrics, desc="Updating metrics", leave=False):
            try:
                metric.update(*args, **kwargs)
            except Exception as e:
                errors.append(f"Error updating metric {metric.name}: {e}")
        for error in errors:
            print(error)

    def compute(self) -> Dict[str, float]:
        """
        Computes and returns the results of all metrics.

        Returns:
            Dict[str, float]: A dictionary mapping metric names to their computed results.
        """
        return {metric.name: metric.compute() for metric in self.metrics}

    def reset(self) -> None:
        """
        Resets all metrics.
        """
        for metric in self.metrics:
            metric.reset()

    def __getitem__(self, item: str):
        """
        Retrieves a metric by its name.

        Args:
            item (str): The name of the metric to retrieve.

        Returns:
            BaseMetric: The metric instance.

        Raises:
            KeyError: If the metric is not found.
        """
        for metric in self.metrics:
            if metric.name == item:
                return metric
        raise KeyError(f"Metric {item} not found")

    def to_dict(self) -> Dict[str, float]:
        """
        Returns the results of all metrics as a dictionary.

        Returns:
            Dict[str, float]: A dictionary mapping metric names to their computed results.
        """
        return {metric.name: metric.compute() for metric in self.metrics}


