Optimizers
==========

This module provides a framework for integrating PyTorch optimizers with a configurable schema, allowing dynamic configuration and extension of optimizers.

.. currentmodule:: optimizer

Adding a Custom Optimizer
-------------------------

To add a custom optimizer, you need to define a new class that extends the ``BaseOptimizer`` class. You can configure the optimizer using a schema and implement the optimization logic in the ``step`` method.

Registering Optimizers
----------------------

The module automatically registers all valid optimizer classes from ``torch.optim`` as subclasses of ``BaseOptimizer``. This allows for seamless integration and configuration of PyTorch's built-in optimizers.

For more information on the available optimizers and their parameters, refer to the official PyTorch documentation: `torch.optim <https://pytorch.org/docs/stable/optim.html>`_.

Configuration Schema Generation
*******************************

The module provides functionality to automatically generate a configuration schema for a given optimizer by inspecting its ``__init__`` parameters. This allows for dynamic and flexible configuration of optimizers.

.. autofunction:: src.optimizer.generate_config_schema

.. autofunction:: src.optimizer.infer_type_from_default

BaseOptimizer
-------------

.. autoclass:: src.optimizer.BaseOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseOptimizer`` class serves as the base class for integrating PyTorch optimizers with the ``TypedConfigurable`` framework. It enables dynamic subclass generation for each optimizer, allowing for easy configuration and extension.
