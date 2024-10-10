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
        self.name = 'accuracy'

    def update(self, pred, target, missing_data=None, **kwargs):
        if missing_data is not None:
            idx = missing_data > 0
            pred = pred[idx]
            target = target[idx]

        correct = torch.sum(pred == target)
        total = pred.numel()

        self.correct += correct.item()
        self.total += total



class Dice(Metric):
    config_schema = {'n_thresholds': Schema(int, default=4)}
    aliases = ['dice', 'dice_index']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'dice'
        self.thresholds = [(i + 1) * 0.2 for i in range(self.n_thresholds)]
        self.dice_t = np.zeros_like(self.thresholds)
        print(self.dice_t)


    def update(self, pred, target, missing=None, **kwargs):
        for i, threshold in enumerate(self.thresholds):
            segmentation = (pred >= threshold).type(torch.int)
            dice = self._core(segmentation, target, missing)
            self.dice_t[i] += dice
            self.total += 1

    def compute(self):
        return self.dice_t / self.count

    def reset(self):
        if hasattr(self, 'dice'):
            del self.dice
        self.count = 0

    def _core(self, preds, targets, missing_data=None):
        if missing_data is not None:
            idx = missing_data > 0
            preds = preds[idx]
            targets = targets[idx]

        segData_TP = preds + targets
        TP_value = 2
        TP = (segData_TP == TP_value).sum().item()
        segData_FP = 2 * preds + targets
        segData_FN = preds + 2 * targets
        FP = (segData_FP == 2).sum().item()
        FN = (segData_FN == 2).sum().item()
        if 2 * TP + FP + FN > 0:
            return 2 * TP / (2 * TP + FP + FN)
        return 1.0


class Segmentation_acc(Metric):
    aliases = ['segmentation_acc', 'seg_acc']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'segmentation_acc'

    def update(self, pred, target, **kwargs):
        nb = target.sum().item()
        if nb > 0:
            correct = (pred * target).sum().item()
            self.correct += correct
            self.total += nb
