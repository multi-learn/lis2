Metrics
==============

This module provides a collection of metrics for evaluating model performance. Each metric is a subclass of ``BaseMetric`` and can be used to compute specific evaluation scores.

.. currentmodule:: metrics

BaseMetric Class
----------------

.. autoclass:: lis2.metrics.base_metric.BaseMetric
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseMetric`` class is an abstract base class for all metrics. It provides methods for updating, computing, and resetting metric results.

Example:

.. code-block:: python

    class CustomMetric(BaseMetric):
        def __init__(self):
            super().__init__()
            self.name = "custom_metric"

        def update(self, pred, target, idx):
            # Custom update logic
            score = np.sum((pred == target) * idx)
            self.result += score
            self.averaging_coef += idx.sum()

        def compute(self):
            return super().compute()

    # Usage
    metric = CustomMetric()
    metric.update(np.array([0, 1, 0]), np.array([0, 1, 1]), np.array([1, 1, 0]))
    print(metric.compute())  # Outputs the computed metric result

Metrics Zoo
-----------

- The ``MSSIM`` class uses the structural similarity function from ``skimage.metrics``.
- The ``AveragePrecision`` class uses the ``average_precision_score`` function from ``sklearn.metrics``.
- The ``ROCAUCScore`` class uses the ``ROCAUCScore_score`` function from ``sklearn.metrics``.
- The ``Dice`` class uses custom logic to compute the Dice coefficient.
- All metrics inherit from ``BaseMetric`` and follow a similar structure for updating and computing results.


MSSIM Metric
************

.. autoclass:: lis2.metrics.mssim.MSSIM
   :members:
   :undoc-members:
   :show-inheritance:

The ``MSSIM`` class computes the Mean Structural Similarity Index (MSSIM) between model predictions and ground truth. It uses a threshold to binarize predictions and calculates the structural similarity for each segmentation.

Average Precision Metric
************************

.. autoclass:: lis2.metrics.average_precision.AveragePrecision
   :members:
   :undoc-members:
   :show-inheritance:

The ``AveragePrecision`` class computes the average precision score between model predictions and ground truth. It rounds predictions to create binary masks and uses the average precision score from scikit-learn.

ROC AUC Metric
**************

.. autoclass:: lis2.metrics.roc.ROCAUCScore
   :members:
   :undoc-members:
   :show-inheritance:

The ``ROCAUCScore`` class computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for evaluating binary classification models.

Dice Metric
***********

.. autoclass:: lis2.metrics.dice.Dice
   :members:
   :undoc-members:
   :show-inheritance:

The ``Dice`` class computes the Dice coefficient, a measure of overlap between two samples. It is often used for evaluating segmentation models.

Metric Manager
--------------

.. autoclass:: lis2.metrics.metric_manager.MetricManager
   :members:
   :undoc-members:
   :show-inheritance:

The ``MetricManager`` class manages multiple metrics, allowing for batch updates, computations, and resets. It provides a convenient way to handle multiple evaluation metrics simultaneously.

