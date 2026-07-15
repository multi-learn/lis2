import inspect

import torch.optim as optim
from configurable import TypedConfigurable, Schema
from torch.optim import Optimizer

from .utils import generate_config_schema


def register_optimizers() -> None:
    """
    Registers all valid optimizer classes from torch.optim as subclasses of BaseOptimizer.
    """
    optimizer_classes = inspect.getmembers(optim, inspect.isclass)
    for name, cls in optimizer_classes:
        if issubclass(cls, Optimizer) and cls is not Optimizer:
            subclass = type(
                name,
                (BaseOptimizer, cls),
                {
                    "__module__": __name__,
                    "aliases": [name.lower()],
                    "config_schema": generate_config_schema(cls),
                },
            )
            globals()[name] = subclass


class BaseOptimizer(TypedConfigurable, Optimizer):
    """
    Base class for PyTorch optimizers integrated with TypedConfigurable.

    Enables dynamic subclass generation for each optimizer.
    Check torch.optim documentation for more information on how to implement custom optimizers.

    Configuration:
        - **lr** (float): The learning rate for the optimizer.

    Example:
        Here's how you can create and use a custom optimizer by inheriting from `BaseOptimizer`:

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

    """

    config_schema = {"lr": Schema(float, default=0.01)}
    pass
