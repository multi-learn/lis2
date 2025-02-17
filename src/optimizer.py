import inspect
from typing import List, Tuple, Any, Type, Dict

import torch.optim as optim
from configurable import TypedConfigurable, Schema
from torch.optim import Optimizer


def generate_config_schema(optimizer_class: Type[Optimizer]) -> Dict[str, Schema]:
    """
    Automatically generates a configuration schema for a given optimizer
    by inspecting its __init__ parameters.

    Args:
        optimizer_class (Type[Optimizer]): The optimizer class to generate the schema for.

    Returns:
        Dict[str, Schema]: A dictionary mapping parameter names to their corresponding schema.
    """
    config_schema = {}
    init_signature = inspect.signature(optimizer_class.__init__)
    for param_name, param in init_signature.parameters.items():
        if param_name in ["self", "params", "defaults"]:
            continue

        # Determine if the parameter is optional
        optional = param.default != inspect.Parameter.empty
        default = param.default if optional else None

        # Infer the type
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation
        elif param.default != inspect.Parameter.empty:
            param_type = infer_type_from_default(param.default)
        else:
            param_type = str  # Default to string if no annotation or default

        config_schema[param_name] = Schema(
            type=param_type,
            optional=optional,
            default=default,
        )
    return config_schema


def infer_type_from_default(default_value: Any) -> Any:
    """
    Infers the type annotation from the default value.

    Args:
        default_value (Any): The default value to infer the type from.

    Returns:
        Any: The inferred type.
    """
    if isinstance(default_value, tuple):
        # Infer types of elements in the tuple
        element_types = tuple(type(element) for element in default_value)
        # Create a typing.Tuple with the inferred element types
        return Tuple[element_types]
    elif isinstance(default_value, list):
        # Infer the type of elements in the list
        element_type = type(default_value[0]) if default_value else Any
        return List[element_type]
    else:
        return type(default_value)


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
                }
            )
            globals()[name] = subclass

class BaseOptimizer(TypedConfigurable, Optimizer):
    """
    Base class for PyTorch optimizers integrated with TypedConfigurable.

    Enables dynamic subclass generation for each optimizer.
    Check torch.optim documentation for more information on how to implement custom optimizers.

    **Configuration**:

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
