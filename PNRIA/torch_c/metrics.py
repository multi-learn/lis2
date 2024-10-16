import abc
import numpy as np
from PNRIA.configs.config import TypedCustomizable, Schema

import torch


class Metrics:
    def __init__(self, metrics_configs):
        self.metrics = []
        for metric_config in metrics_configs:
            metric = Metric.from_config(metric_config)
            self.metrics.append(metric)

    def update(self, *args, **kwargs):
        for metric in self.metrics:
            metric.update(*args, **kwargs)

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
        raise KeyError('Metric {} not found'.format(item))

    def to_dict(self):
        return {metric.name: metric.compute() for metric in self.metrics}


class Metric(abc.ABC, TypedCustomizable):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reset()
        self.correct = 0
        self.total = 0

    @abc.abstractmethod
    def update(self, pred, target, *args, **kwargs):
        pass

    def compute(self):
        if self.total > 0:
            return self.correct / self.total
        raise ValueError('No data to compute metric: {}'.format(self.name))

    def reset(self):
        self.correct = 0
        self.total = 0


class Accuracy(Metric):
    aliases = ['accuracy', 'acc']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'positive_part_accuracy'
        self.averaging_coef = 0  # Accumulateur pour les pixels valides

    def update(self, pred, target, missing=None, **kwargs):
        if missing is not None:
            idx = missing > 0
            pred = pred[idx]
            target = target[idx]
            coef = idx.sum().item()  # Nombre de pixels valides
        else:
            coef = target.numel()  # Nombre total de pixels

        # Convertir les prédictions en binaires
        pred = torch.round(pred)

        # Calculer les pixels corrects
        correct = torch.sum(pred == target).item()

        # Accumuler les résultats et le coefficient
        self.correct += correct
        self.total += coef
        self.averaging_coef += coef

    def compute(self):
        # Normaliser la précision avec le coefficient
        if self.averaging_coef > 0:
            return self.correct / self.averaging_coef
        return 0.0


class Dice(Metric):
    config_schema = {'n_thresholds': Schema(int, default=4)}
    aliases = ['dice', 'dice_index']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'dice'
        self.thresholds = [(i + 1) * 0.2 for i in range(self.n_thresholds)]
        self.dice_t = np.zeros_like(self.thresholds)
        self.averaging_coef = 0  # Accumulateur pour les pixels valides

    def update(self, pred, target, missing=None, **kwargs):
        for i, threshold in enumerate(self.thresholds):
            segmentation = (pred >= threshold).int()

            if missing is not None:
                idx = missing > 0
                segmentation = segmentation[idx]
                target = target[idx]
                coef = idx.sum().item()  # Nombre de pixels valides
            else:
                coef = target.numel()

            dice = self._core(segmentation, target)
            self.dice_t[i] += dice * coef  # Pondérer par le nombre de pixels valides
            self.averaging_coef += coef

    def compute(self):
        if self.averaging_coef > 0:
            return self.dice_t / self.averaging_coef
        return self.dice_t

    def _core(self, preds, targets):
        TP = (preds * targets).sum().item()
        FP = preds.sum().item() - TP
        FN = targets.sum().item() - TP

        if 2 * TP + FP + FN > 0:
            return 2 * TP / (2 * TP + FP + FN)
        return 1.0


class Segmentation_acc(Metric):
    aliases = ['segmentation_acc', 'seg_acc']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'segmentation_acc'
        self.averaging_coef = 0  # Accumulateur pour les pixels positifs

    def update(self, pred, target, missing=None, **kwargs):
        if missing is not None:
            target = missing * target

        total_positive = target.sum().item()  # Nombre total de pixels positifs
        correct = (pred * target).sum().item()  # Nombre de prédictions correctes

        self.correct += correct
        self.total += total_positive
        self.averaging_coef += total_positive  # Ajouter le coefficient basé sur les pixels positifs

    def compute(self):
        if self.averaging_coef > 0:
            return self.correct / self.averaging_coef
        return 0.0