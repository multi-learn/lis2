import abc
import inspect
import warnings
from enum import Enum
from typing import List, Dict, Any, Tuple

import torch
import torchvision.transforms.v2 as transforms
from configurable import TypedConfigurable, Schema
from torch.fft import Tensor


class DataAugmentations:
    """
    This class acts as a "Controller" to apply a list of data augmentations to given data.

    For each element of the data, the augmentations are applied in the order they are specified in the list.
    The class is also checking the `keys_to_augment` value in each configuration, to ensure that only the data
    corresponding to the specified keys is augmented.

    Notes:
        - The configuration should include a ``ToTensor`` augmentation to ensure that the final output is a Tensor.
        - If no ``ToTensor`` augmentation is provided by the user, the module automatically **appends a default ``ToTensor`` transform** to the configuration.
        - Although there is no strict constraint regarding the order of augmentations, care must be taken when adding new augmentations. The placement of the ``ToTensor`` transform should be handled similarly to a Torch Compose (see: `torchvision.transforms.Compose <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose>`_) to avoid type errors.
        - **Note:** The input to the augmentation pipeline is expected to be NumPy arrays, while the final output must be a ``torch.Tensor``.
    """

    def __init__(self, augmentations_configs: List[Dict[str, Any]]):
        """
        Initialize the DataAugmentations class.

        Args:
            augmentations_configs (List[Dict[str, Any]]): A list of dictionaries containing the configuration of the augmentations to apply.
        """
        if not isinstance(augmentations_configs, list):
            raise ValueError("augmentations_configs must be a list")

        if not augmentations_configs:
            warnings.warn(
                "Empty list detected. Adding default ToTensor augmentation.",
                UserWarning,
            )
            has_to_tensor = False
        else:
            has_to_tensor = any(
                config.get("type") == "ToTensor" for config in augmentations_configs
            )
        self.augmentations_configs = augmentations_configs
        if not has_to_tensor:
            augmentations_configs.append({"type": "ToTensor"})
        self.augmentations = [
            BaseDataAugmentation.from_config(config) for config in augmentations_configs
        ]

    def compute(self, data):
        for index, augmentation in enumerate(self.augmentations):
            data = augmentation(data)

        assert all([isinstance(v, (Tensor, List[Tensor])) for _, v in data.items()]), (
            f"End of augmentation pipeline must return Tensor(s). Got {[type(v) for v in data.values()]}.\n"
            'Add a ToTensor augmentation ("type": "ToTensor") at the end of the pipeline.'
        )

        return data


def register_transforms() -> None:
    """
    Registers all valid transforms classes from torch.optim as subclasses of BaseDataAugmentation.
    """
    EXCLUDE_TRANSFORMS = [
        "compose",
        "Compose",
        "Lambda",
        "UniformTemporalSubsample",
        "ToTensor",
        "SanitizeBoundingBoxes",
        "TrivialAugmentWide",
    ]
    transforms_classes = inspect.getmembers(transforms, inspect.isclass)
    EXCLUDE_TRANSFORMS.extend(
        name
        for name, _ in transforms_classes
        if any(sub in name.lower() for sub in ["random", "rand"])
    )
    for name, cls in transforms_classes:
        if (
            name in EXCLUDE_TRANSFORMS
            or isinstance(cls, Enum)
            or issubclass(cls, Enum)
            or issubclass(cls, torch.Tensor)
        ):
            continue
        else:
            transform_method = getattr(cls, "__call__", None)
            if transform_method:

                def transform_fn(self, *args, **kwargs):
                    return transform_method(self, *args, **kwargs)

            else:
                raise ValueError(
                    f"Transform class {cls} does not have a __call__ method."
                )

            subclass = type(
                name,
                (BaseDataAugmentationWithKeys, cls),
                {
                    "__module__": __name__,
                    "aliases": [name.lower()],
                    "config_schema": generate_config_schema(cls),
                    "transform": transform_fn,
                },
            )

        globals()[name] = subclass


def generate_config_schema(transform_class) -> Dict[str, Schema]:
    """
    Automatically generates a configuration schema for a given transform
    by inspecting its __init__ parameters.

    Args:
        transform_class : The transform to generate the schema for.

    Returns:
        Dict[str, Schema]: A dictionary mapping parameter names to their corresponding schema.
    """
    config_schema = {}
    init_signature = inspect.signature(transform_class.__init__)
    for param_name, param in init_signature.parameters.items():
        if param_name in ["self", "params", "defaults", "name"]:
            continue

        optional = param.default != inspect.Parameter.empty
        default = param.default if optional else None

        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation
        elif param.default != inspect.Parameter.empty:
            param_type = infer_type_from_default(param.default)
        else:
            param_type = Any

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
        element_types = tuple(type(element) for element in default_value)
        return Tuple[element_types]
    elif isinstance(default_value, list):
        element_type = type(default_value[0]) if default_value else Any
        return List[element_type]
    else:
        return type(default_value)


class BaseDataAugmentation(abc.ABC, TypedConfigurable):
    """
    BaseDataAugmentation is an abstract class that extends TypedConfigurable
    """

    pass

    def __call__(self, data):
        return self.transform(data)

    @abc.abstractmethod
    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass


class BaseDataAugmentationWithKeys(BaseDataAugmentation, abc.ABC):
    """
    Abstract base class for data augmentation techniques that apply to specific keys in a dataset.

    This class is designed for augmentation techniques that require a list of dataset keys. If the `keys_to_augment`
    list is provided, augmentation will be applied only to those specific keys. If the list is empty, the augmentation
    will be applied to all keys in the dataset.

    Configuration:
        - **keys_to_augment** (List[str]): List of keys in the dataset to apply augmentation (default: []).
          **If empty, the augmentation is applied to all keys.**

    Methods:
        - __call__: Calls the transformation function to apply the augmentation.
        - transformation: Abstract method to be implemented by subclasses to apply the actual transformation.

    Example Configuration (YAML):
        .. code-block:: yaml

            keys_to_augment: ["patch", "spines"]
    """

    config_schema = {
        "keys_to_augment": Schema(List[str], default=[]),
    }

    def __call__(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Applies the data augmentation transformation to the specified keys.

        Verifies that all keys in `keys_to_augment` are present in the `data`.
        If `keys_to_augment` is empty, augmentation is applied to all keys.

        Args:
            data (dict[str, Tensor]): Input dictionary containing tensors.

        Returns:
            dict[str, Tensor]: Dictionary with transformed tensors for the specified keys.
        """
        missing_keys = [key for key in self.keys_to_augment if key not in data]
        if missing_keys:
            raise KeyError(f"Missing keys in the data: {', '.join(missing_keys)}")

        keys = list(data.keys())
        if self.keys_to_augment:
            keys = self.keys_to_augment if self.keys_to_augment else list(data.keys())

        data_filtered = {key: data[key] for key in keys}

        transformed = self.transform(data_filtered)

        result = data.copy()
        for key in keys:
            result[key] = transformed[key]

        return result

    @abc.abstractmethod
    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply the augmentation to the dataset.

        This method must be implemented by subclasses to define the specific transformation logic.

        Args:
            data (dict[str, Tensor]): A dictionary where keys are dataset identifiers (strings) and values are tensors.
            The dictionary will contain keys specified in `keys_to_augment`. These keys must be present in the
            dataset, which is provided by the `Dataset` class. If `keys_to_augment` is empty, augmentation will
            be applied to all the keys in `data`.

        Returns:
            List[Tensor]: A list or dictionary of tensors with applied augmentation, depending on the transformation.
        """
        pass


class NoiseDataAugmentation(BaseDataAugmentationWithKeys):
    """
    Applies random Gaussian noise to dataset patches for data augmentation.

    This class applies random Gaussian noise to the specified dataset patches. The level of noise
    is controlled by the `noise_var` parameter, which defines the variance of the noise. Noise is applied
    only to the keys listed in the `keys_to_augment` configuration.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique (required).
        - **keys_to_augment** (List[str]): List of keys in the dataset to apply the augmentation to (default: [] => all),
          e.g., ["patch", "spines"].
        - **noise_var** (float): Variance of the Gaussian noise to be applied (default: 0).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "NoiseDataAugmentation"
            name: "GaussianNoise"
            keys_to_augment: ["patch", "spines"]
            noise_var: 0.1
    """

    config_schema = {
        "keys_to_augment": Schema(List[str], default=[]),
        "noise_var": Schema(float, default=0),
    }

    def transform(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Applies random noise to the tensors in the provided dictionary, only to the keys
        specified in `self.keys_to_augment`. Ensures that all tensors have the same size
        before applying the noise.

        Args:
            data (dict[str, Tensor]): A dictionary where keys are string identifiers,
                                       and values are tensors.

        Returns:
            dict[str, Tensor]: A dictionary with the same keys, but the tensors are
                                updated with noise applied to the specified keys.
        """
        tensor_sizes = {value.size() for value in data.values()}
        assert (
            len(tensor_sizes) == 1
        ), "All tensors must have the same size to apply the same noise."

        noise = torch.randn_like(next(iter(data.values()))) * self.noise_var

        new_data = {
            key: torch.clamp(value + noise, 0.0, 1.0) for key, value in data.items()
        }

        return new_data
