import abc
from typing import List, Dict, Any

import numpy as np
from configurable import TypedConfigurable, Schema
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm


class Metrics:
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


class BaseMetric(abc.ABC, TypedConfigurable):
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
    def update(self, pred: np.ndarray, target: np.ndarray, idx: np.ndarray, *args: Any, **kwargs: Any) -> None:
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


class AveragePrecision(BaseMetric):
    aliases = ["average_precision", "ap", "map"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the AveragePrecision metric.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "average_precision"

    def update(self, pred: np.ndarray, target: np.ndarray, idx: np.ndarray, *args: Any, **kwargs: Any) -> None:
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

        segData_TP = pred + target
        TP_value = 2
        TP = (segData_TP == TP_value).sum().item()
        segData_FP = 2 * pred + target
        segData_FN = pred + 2 * target
        FP = (segData_FP == 2).sum().item()
        FN = (segData_FN == 2).sum().item()
        if 2 * TP + FP + FN > 0:
            return 2 * TP / (2 * TP + FP + FN)
        return 1.0


class ROCAUCScore(BaseMetric):
    aliases = ["roc_auc", "auc"]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the ROCAUCScore metric.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "roc_auc"

    def update(self, pred: np.ndarray, target: np.ndarray, idx: np.ndarray, *args: Any, **kwargs: Any) -> None:
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


class MSSIM(BaseMetric):
    config_schema = {"win_size": Schema(int, default=7)}
    aliases = ["mssim", "ssim"]

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        """
        Initializes the MSSIM metric.

        Args:
            threshold (float): Threshold for binarizing predictions.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(**kwargs)
        self.name = "mean_ssim"
        self.threshold = threshold

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

    def update(self, pred: np.ndarray, target: np.ndarray, idx: np.ndarray, *args: Any, **kwargs: Any) -> None:
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
