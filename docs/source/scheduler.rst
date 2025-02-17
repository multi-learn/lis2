Learning Rate Schedulers
========================

This module provides a framework for integrating PyTorch learning rate schedulers with a configurable schema, allowing dynamic configuration and extension of schedulers.

.. currentmodule:: scheduler

Adding a Custom Scheduler
-------------------------

To add a custom scheduler, you need to define a new class that extends the ``BaseScheduler`` class. You can configure the scheduler using a schema and implement the learning rate update logic in the ``get_lr`` method.


BaseScheduler
-------------------

.. autoclass:: BaseScheduler
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseScheduler`` class serves as the base class for integrating PyTorch learning rate schedulers with the ``TypedConfigurable`` framework. It enables dynamic subclass generation for each scheduler, allowing for easy configuration and extension.

**Example:**

Here's how you can create and use a custom scheduler by inheriting from ``BaseScheduler``:

.. code-block:: python

    import torch
    from torch.optim.lr_scheduler import _LRScheduler

    class MyCustomScheduler(BaseScheduler):

        schema = {
            "step_size": Schema(int, optional=True, default=30),
            "gamma": Schema(float, optional=True, default=0.1),
        }

        def __init__(self, optimizer: Any, last_epoch: int = -1) -> None:
            super().__init__(optimizer, last_epoch)

        def get_lr(self) -> list:
            # Implementation of the learning rate update logic
            return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                    for base_lr in self.base_lrs]

    # Usage example
    model = MyModel()  # Your PyTorch model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = MyCustomScheduler(optimizer, step_size=30, gamma=0.1)

Registering Schedulers
----------------------

The module automatically registers all valid scheduler classes from ``torch.optim.lr_scheduler`` as subclasses of ``BaseScheduler``. This allows for seamless integration and configuration of PyTorch's built-in schedulers.

For more information on the available schedulers and their parameters, refer to the official PyTorch documentation: `torch.optim.lr_scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.

Configuration Schema Generation
-------------------------------

The module provides functionality to automatically generate a configuration schema for a given scheduler by inspecting its ``__init__`` parameters. This allows for dynamic and flexible configuration of schedulers.

.. autofunction:: generate_config_schema

Conclusion
----------

This module provides a flexible and extensible framework for integrating and configuring PyTorch learning rate schedulers. By leveraging the ``BaseScheduler`` class and the automatic schema generation, you can easily create and customize schedulers for your specific needs.
