Early Stopping
==============

This module provides implementations for early stopping strategies, which are used to halt training processes when no significant improvement is observed in a monitored metric, such as loss.

.. currentmodule:: early_stop

.. _adding_early_stop:

Adding a Custom Early Stopping Strategy
---------------------------------------

To add a custom early stopping strategy, you need to define a new class that extends the ``BaseEarlyStopping`` class and implements the abstract methods ``step`` and ``reset``.

Here is an example of how to define a custom early stopping class:

.. code-block:: python

    from early_stop import BaseEarlyStopping

    class CustomEarlyStopping(BaseEarlyStopping):
        """
        CustomEarlyStopping is an example implementation extending BaseEarlyStopping.

        This class implements specific logic for monitoring a custom metric.
        """

        def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
            super().__init__(patience, min_delta)
            self.counter = 0
            self.best_metric = None

        def step(self, metric: float) -> bool:
            if self.best_metric is None or metric < self.best_metric - self.min_delta:
                self.best_metric = metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

        def reset(self) -> None:
            self.counter = 0
            self.best_metric = None

After defining the custom class, you can configure it using a YAML configuration file and update the ``__init__.py`` file to make it accessible.



BaseEarlyStopping Class
-----------------------

.. autoclass:: src.early_stop.BaseEarlyStopping
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseEarlyStopping`` class is an abstract base class that defines the interface for early stopping mechanisms. It monitors a metric during training and stops the process if no improvement is observed after a certain number of epochs.

Attributes:

- **patience (int)**: Number of epochs to wait for improvement before stopping.
- **min_delta (float)**: Minimum change in monitored value to qualify as an improvement.

Example usage:

.. code-block:: python

    config = {
        'type': 'loss_early_stopping',
        'patience': 5,
        'min_delta': 0.01
    }
    early_stopper = LossEarlyStopping.from_config(config)

    for epoch in range(num_epochs):
        loss = compute_loss()
        if early_stopper.step(loss):
            print("Early stopping triggered.")
            break

Early Stop Zoo
--------------

LossEarlyStopping Class
***********************

.. autoclass:: src.early_stop.LossEarlyStopping
   :members:
   :undoc-members:
   :show-inheritance:

