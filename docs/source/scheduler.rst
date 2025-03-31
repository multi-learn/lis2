Learning Rate Schedulers
========================

This module provides a framework for integrating PyTorch learning rate schedulers with a configurable schema, allowing dynamic configuration and extension of schedulers.

.. currentmodule:: scheduler

.. _adding_scheduler:

Adding a Custom Scheduler
-------------------------

To add a custom scheduler, you need to define a new class that extends the ``BaseScheduler`` class. You can configure the scheduler using a schema and implement the learning rate update logic in the ``get_lr`` method.

Registering Schedulers
----------------------

The module automatically registers all valid scheduler classes from ``torch.optim.lr_scheduler`` as subclasses of ``BaseScheduler``. This allows for seamless integration and configuration of PyTorch's built-in schedulers.

For more information on the available schedulers and their parameters, refer to the official PyTorch documentation: `torch.optim.lr_scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.

Configuration Schema Generation
*******************************

The module provides functionality to automatically generate a configuration schema for a given scheduler by inspecting its ``__init__`` parameters. This allows for dynamic and flexible configuration of schedulers.

.. autofunction:: src.scheduler.generate_config_schema


BaseScheduler
-------------

.. autoclass:: src.scheduler.BaseScheduler
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseScheduler`` class serves as the base class for integrating PyTorch learning rate schedulers with the ``TypedConfigurable`` framework. It enables dynamic subclass generation for each scheduler, allowing for easy configuration and extension.

