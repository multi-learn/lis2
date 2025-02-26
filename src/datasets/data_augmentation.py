import abc
import inspect
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from enum import Enum

import numpy as np
import torch
from configurable import TypedConfigurable, Schema
import torchvision.transforms.transforms as transforms
from torchvision.transforms import Compose


class DataAugmentations:
    """
    This class acts as a "Controller" to apply a list of data augmentations to given data.

    For each element of the data, the augmentations are applied in the order they are specified in the list.
    The class is also checking the `keys_to_augment` value in each configuration, to ensure that only the data
    corresponding to the specified keys is augmented.
    """

    def __init__(self, augmentations_configs: List[Dict[str, Any]]):
        """
        Initializes the Augmentations class with a list of augmentation configurations.

        Args:
            augmentations_configs (List[Dict[str, Any]]): List of augmentations configurations.
        """
        if isinstance(augmentations_configs, list) and augmentations_configs:
            self.augmentations_configs = augmentations_configs
            self.verify_augmentations()
            self.to_augment = [
                config.get("keys_to_augment", []) for config in augmentations_configs
            ]
            self.augmentations = [
                BaseDataAugmentation.from_config(config)
                for config in augmentations_configs
            ]
        else:
            raise (
                ValueError(
                    "augmentations_configs must be a non empty list of augmentations configurations"
                )
            )

    def verify_augmentations(self):
        """For performances reasons, ExtendedDataAugmentation is computed in numpy and not torch, thus must be performed first"""
        if any(
            a["type"] == "ExtendedDataAugmentation" for a in self.augmentations_configs
        ):
            assert (
                self.augmentations_configs[0]["type"] == "ExtendedDataAugmentation"
            ), "If ExtendedDataAugmentation is specified, it must be the first augmentation"
            assert (
                len(self.augmentations_configs) > 1
                and self.augmentations_configs[1]["type"] == "ToTensor"
            ), "If ExtendedDataAugmentation in the list, ToTensor must be the second augmentation"
            assert (
                "keys_to_augment" not in self.augmentations_configs[1]
                or self.augmentations_configs[1]["keys_to_augment"] == []
            ), "ToTensor must have no `keys to augment` to be applied to all the data"
        else:
            assert (
                self.augmentations_configs[0]["type"] == "ToTensor"
            ), "ToTensor must be the first augmentation"
            assert (
                "keys_to_augment" not in self.augmentations_configs[0]
                or self.augmentations_configs[0]["keys_to_augment"] == []
            ), "ToTensor must have no `keys to augment` to be applied to all the data"

    def compute(self, data):
        for index, augmentation in enumerate(self.augmentations):
            # To augment is empty, meaning we augment every elements of the data
            if self.to_augment[index]:
                for k, v in data.items():
                    if k in self.to_augment[index]:
                        data[k] = augmentation(v)
                    else:
                        data[k] = v
            else:
                for k, v in data.items():
                    data[k] = augmentation(v)

        return list(data.values())


def register_transforms() -> None:
    """
    Registers all valid transforms classes from torch.optim as subclasses of BaseDataAugmentation.
    """
    EXCLUDE_TRANSFORMS = ["compose"]
    transforms_classes = inspect.getmembers(transforms, inspect.isclass)
    for name, cls in transforms_classes:
        if (
            name == "Compose"
            or isinstance(cls, Enum)
            or issubclass(cls, Enum)
            or issubclass(cls, torch.Tensor)
        ):
            continue
        else:
            subclass = type(
                name,
                (BaseDataAugmentation, cls),
                {
                    "__module__": __name__,
                    "aliases": [name.lower()],
                    "config_schema": generate_config_schema(cls),
                },
            )
        globals()[name] = subclass


def generate_config_schema(transform_class) -> Dict[str, Schema]:
    """
    Automatically generates a configuration schema for a given transform
    by inspecting its __init__ parameters.

    Args:
        transform_class (Type[Optimizer]): The optimizer class to generate the schema for.

    Returns:
        Dict[str, Schema]: A dictionary mapping parameter names to their corresponding schema.
    """
    config_schema = {}
    init_signature = inspect.signature(transform_class.__init__)
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
            param_type = Any  # Default to string if no annotation or default

        config_schema[param_name] = Schema(
            type=param_type,
            optional=optional,
            default=default,
        )

    config_schema["keys_to_augment"] = Schema(type=List[str], default=[], optional=True)
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


class BaseDataAugmentation(abc.ABC, TypedConfigurable):
    pass


class NoiseDataAugmentation(BaseDataAugmentation):
    """
    NoiseDataAugmentation for adding random noise to dataset patches.

    This data augmentation class applies random noise to dataset patches. The noise level
    is controlled by the ``noise_var`` parameter.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique (required).
        - **keys_to_augment** (List[str]): List of keys in the dataset to apply augmentation (default: []).
        - **noise_var** (float): Variance of the Gaussian noise to be applied (default: 0).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "NoiseDataAugmentation"
            name: "GaussianNoise"
            keys_to_augment: ["patch", "spines"]
            noise_var: 0.1
    """

    config_schema = {
        "type": Schema(str),
        "name": Schema(str),
        "keys_to_augment": Schema(List[str], default=[]),
        "noise_var": Schema(float, default=0),
    }

    def __call__(self, data):
        return self.apply_data_augmentation(data)

    def apply_data_augmentation(self, data):
        """
        Apply a data augmentation scheme to given data

        Parameters
        ----------
        data: list[torch.Tensor]
            A list of patches (input, output, others...)

        Returns
        -------
        The list of transformed patches in the same order as input.
        """
        new_data = self.transformation(data)

        new_data = torch.clamp(new_data, 0.0, 1.0)

        return new_data

    def transformation(self, data):
        """
        Apply a random transform to a tensor

        Parameters
        ----------
        data: torch.Tensor

        Returns
        -------
        Transformed Tensor
        """
        noise = torch.randn_like(data) * self.noise_var

        res = data + noise

        return res


class ExtendedDataAugmentation(BaseDataAugmentation):
    """
    ExtendedDataAugmentation for applying advanced data augmentation techniques.

    This class extends data augmentation functionalities by allowing the application of
    various transformations to dataset patches. The augmentation is controlled by configurable
    parameters, including `keys_to_augment` and `noise_var`, which define which dataset components
    to modify and the level of noise to apply.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique (required).
        - **keys_to_augment** (List[str]): List of keys in the dataset to apply augmentation (default: []).
        - **noise_var** (float): Variance of the Gaussian noise to be applied (default: 0).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "ExtendedDataAugmentation"
            name: "AdvancedAugmentation"
            keys_to_augment: ["patch", "spines"]
            noise_var: 0.05
    """

    config_schema = {
        "type": Schema(str),
        "name": Schema(str),
        "keys_to_augment": Schema(List[str], default=[]),
        "noise_var": Schema(float, default=0),
    }

    def __call__(self, data):
        return self.apply_data_augmentation(data)

    def apply_data_augmentation(self, data):
        """
        Apply a data augmentation scheme to given data

        Parameters
        ----------
        data: list[np.ndarray]
            A list of patches (input, output, others...)

        Returns
        -------
        The list of transformed patches in the same order as input.
        """

        new_data = self.transformation(data=data, noise=self.noise_var)
        new_data = np.clip(np.array(new_data, dtype="f"), 0.0, 1.0)

        return new_data

    def transformation(self, data, noise):
        """
        Apply a random transform to a list of data with noise everywhere (extended version)
        Parameters
        ----------
        data_list: list
            A list of data
        rng: random.Random
            A random generator
        noise_var: list[float]
            The variance of the additional noise for each element

        Returns
        -------
        A list of transform data
        """
        self.random_gen = random.Random()
        n_tf = self.random_gen.randint(0, 15)
        noise = np.array(np.random.standard_normal(data.shape) * noise, dtype="f")
        data = data + noise
        res = self.random_augment(data, n_tf)
        return res

    def random_augment(self, data, num_tf):
        """
        Transform the data using a given transformation

        Parameters
        ----------
        data: numpy.ndarray
            The data to transform
        num_tf: int
            The number of the transformation

        Returns
        -------
        The transformed data
        """
        shape = data.shape
        data = np.squeeze(data)

        if num_tf == 1:
            data = np.fliplr(data)
        elif num_tf == 2:
            data = np.flipud(data)
        elif num_tf == 3:
            data = np.rot90(data)
        elif num_tf == 4:
            data = np.rot90(data, 2)
        elif num_tf == 5:
            data = np.rot90(data, 3)
        elif num_tf == 6:
            data = np.fliplr(np.flipud(data))
        elif num_tf == 7:
            data = np.fliplr(np.rot90(data))
        elif num_tf == 8:
            data = np.fliplr(np.rot90(data, 2))
        elif num_tf == 9:
            data = np.fliplr(np.rot90(data, 3))
        elif num_tf == 10:
            data = np.flipud(np.rot90(data))
        elif num_tf == 11:
            data = np.flipud(np.rot90(data, 2))
        elif num_tf == 12:
            data = np.flipud(np.rot90(data, 3))
        elif num_tf == 13:
            data = np.fliplr(np.flipud(np.rot90(data)))
        elif num_tf == 14:
            data = np.fliplr(np.flipud(np.rot90(data, 2)))
        elif num_tf == 15:
            data = np.fliplr(np.flipud(np.rot90(data, 3)))

        return np.reshape(data.copy(), shape)
