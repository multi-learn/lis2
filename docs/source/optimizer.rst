Optimizers
==========

This module provides a framework for integrating PyTorch optimizers with a configurable schema, allowing dynamic configuration and extension of optimizers.

.. currentmodule:: optimizer

Adding a Custom Optimizer
-------------------------

To add a custom optimizer, you need to define a new class that extends the ``BaseOptimizer`` class. You can configure the optimizer using a schema and implement the optimization logic in the ``step`` method.


BaseOptimizer
-------------

.. autoclass:: src.optimizer.BaseOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseOptimizer`` class serves as the base class for integrating PyTorch optimizers with the ``TypedConfigurable`` framework. It enables dynamic subclass generation for each optimizer, allowing for easy configuration and extension.

**Example:**

Here's how you can create and use a custom optimizer by inheriting from ``BaseOptimizer``:

.. code-block:: python

    import torch
    from torch.optim.optimizer import Optimizer

    class MyCustomOptimizer(BaseOptimizer):

        schema = {
            "lr": Schema(float, optional=True, default=0.01),
        }

        def __init__(self, params):
            defaults = {"lr": self.lr}
            super().__init__(params, defaults)

        def step(self, closure=None):
            # Implementation of the optimization step
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    p.data.add_(-group['lr'], d_p)

    # Usage example
    model = MyModel()  # Your PyTorch model
    optimizer = MyCustomOptimizer(model.parameters(), lr=0.01)

Registering Optimizers
----------------------

The module automatically registers all valid optimizer classes from ``torch.optim`` as subclasses of ``BaseOptimizer``. This allows for seamless integration and configuration of PyTorch's built-in optimizers.

For more information on the available optimizers and their parameters, refer to the official PyTorch documentation: `torch.optim <https://pytorch.org/docs/stable/optim.html>`_.

Configuration Schema Generation
-------------------------------

The module provides functionality to automatically generate a configuration schema for a given optimizer by inspecting its ``__init__`` parameters. This allows for dynamic and flexible configuration of optimizers.

.. autofunction:: src.optimizer.generate_config_schema

.. autofunction:: src.optimizer.infer_type_from_default

Conclusion
----------

This module provides a flexible and extensible framework for integrating and configuring PyTorch optimizers. By leveraging the ``BaseOptimizer`` class and the automatic schema generation, you can easily create and customize optimizers for your specific needs.
