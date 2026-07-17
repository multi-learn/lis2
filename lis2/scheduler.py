import inspect
from typing import Any

from configurable import TypedConfigurable, Schema
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler

from .utils.schema import generate_config_schema

# List of schedulers to exclude from automatic registration
EXCLUDE_SCHEDULERS = [
    "_LRScheduler",
    "LRScheduler",
    "SequentialLR",
    "ChainedScheduler",
    "LambdaLR",
]


def register_schedulers() -> None:
    """
    Registers all valid scheduler classes from torch.optim.lr_scheduler as subclasses of BaseScheduler.
    """
    scheduler_classes = inspect.getmembers(lr_scheduler, inspect.isclass)
    for name, cls in scheduler_classes:
        if (
            issubclass(cls, LRScheduler)
            and cls is not LRScheduler
            and name not in EXCLUDE_SCHEDULERS
        ):
            subclass = type(
                name,
                (BaseScheduler, cls),
                {
                    "__module__": __name__,
                    "aliases": [name.lower()],
                    "config_schema": generate_config_schema(cls),
                },
            )
            globals()[name] = subclass


class BaseScheduler(TypedConfigurable, LRScheduler):
    """
    Base class for PyTorch schedulers integrated with TypedConfigurable.

    Enables dynamic subclass generation for each scheduler.
    Check torch.optim.lr_scheduler documentation for more information
    on how to implement custom schedulers.

    Example:
        Here's how you can create and use a custom scheduler by inheriting from `BaseScheduler`:

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
    """

    pass
