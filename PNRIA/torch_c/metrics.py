import abc

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from PNRIA.configs.config import TypedCustomizable, Schema


class Metrics:
    def __init__(self, metrics_configs):
        self.metrics = []
        for metric_config in metrics_configs:
            metric = Metric.from_config(metric_config)
            self.metrics.append(metric)

    def update(self, *args, **kwargs):
        errors = []
        for metric in tqdm(self.metrics, desc="Updating metrics", leave=False):
            try:
                metric.update(*args, **kwargs)
            except Exception as e:
                errors.append(f"Error updating metric {metric.name}: {e}")
        for error in errors:
            print(error)

    def compute(self):
        results = {}
        for metric in self.metrics:
            results[metric.name] = metric.compute()
        return results

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __getitem__(self, item):
        for metric in self.metrics:
            if metric.name == item:
                return metric
        raise KeyError("Metric {} not found".format(item))

    def to_dict(self):
        return {metric.name: metric.compute() for metric in self.metrics}


class Metric(abc.ABC, TypedCustomizable):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.result = 0
        self.averaging_coef = 0

    @abc.abstractmethod
    def update(self, pred, target, idx, *args, **kwargs):
        pass

    def compute(self):
        if self.averaging_coef > 0:
            return self.result / self.averaging_coef
        raise ValueError("No data to compute metric: {}".format(self.name))

    def reset(self):
        self.result = 0
        self.averaging_coef = 0

    def __str__(self):
        return self.name


from sklearn.metrics import average_precision_score


class AveragePrecision(Metric):
    aliases = ["average_precision", "ap", "map"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "average_precision"
        self.result = 0.0
        self.averaging_coef = 0.0

    def update(self, pred, target, idx, *args, **kwargs):
        """
        Updates the metric with current predictions and targets.

        - pred: Model predictions, rounded for binary masks
        - target: Ground truth labels (binary)
        - idx: Mask or weighting for valid examples
        """

        if len(set(target)) >= 2:
            pred = np.round(pred).astype(int)
            target = np.round(target).astype(int)

            ap = average_precision_score(target.flatten(), pred.flatten())
            self.result += ap * idx.sum().item()
            self.averaging_coef += idx.sum().item()
        else:
            self.logger.debug("Not enough classes to compute Average Precision score.")


class Dice(Metric):
    config_schema = {"threshold": Schema(float, default=0.5)}
    aliases = ["dice", "dice_index"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "dice"

    def update(self, pred, target, idx, **kwargs):

        if len(set(target)) >= 2:
            segmentation = (pred >= self.threshold).astype(int)
            self.result += self._core(segmentation, target) * idx.sum().item()
            self.averaging_coef += idx.sum().item()
        else:
            self.logger.debug("Not enough classes to compute Dice score.")

    def _core(self, pred, target):
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


class ROCAUCScore(Metric):
    aliases = ["roc_auc", "auc"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "roc_auc"

    def update(self, pred, target, idx, *args, **kwargs):
        """
        Updates the metric with current predictions and targets.

        - pred: Model predictions
        - target: Ground truth labels
        - idx: Mask or weighting for valid examples
        """
        if len(set(target)) >= 2:
            pred = np.round(pred).astype(int).flatten()
            target = np.round(target).astype(int).flatten()
            idx = idx.flatten()
            roc_auc = roc_auc_score(target, pred)
            self.result += roc_auc * idx.sum().item()
            self.averaging_coef += idx.sum().item()
        else:
            self.logger.debug("Not enough classes to compute ROC AUC score.")


class MSSIM(Metric):

    config_schema = {"win_size": Schema(int, default=7)}

    aliases = ["mssim", "ssim"]

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.name = "mean_ssim"
        self.result = 0.0
        self.averaging_coef = 0.0
        self.threshold = threshold

    def _core(self, pred, target):
        """
        Computes the Mean Structural Similarity Index (MSSIM).

        - pred: Model predictions (flattened numpy array)
        - target: Ground truth (flattened numpy array)
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

    def update(self, pred, target, idx, *args, **kwargs):
        """
        Updates the MSSIM metric with current predictions and targets.

        - pred: Model predictions (as probabilities, already flattened numpy arrays)
        - target: Ground truth (already flattened numpy arrays)
        - idx: Mask or weighting for valid examples
        """
        mssim = self._core(pred, target)

        self.result += mssim * idx.sum().item()
        self.averaging_coef += idx.sum().item()
