import abc
import inspect
import random
import warnings
from enum import Enum
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torchvision.transforms.transforms as transforms
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
            warnings.warn("Empty list detected. Adding default ToTensor augmentation.", UserWarning)
            has_to_tensor = False
        else:
            has_to_tensor = any(config.get("type") == "ToTensor" for config in augmentations_configs)
        self.augmentations_configs = augmentations_configs
        if not has_to_tensor:
            augmentations_configs.append({"type": "ToTensor"})
        self.augmentations = [BaseDataAugmentation.from_config(config) for config in augmentations_configs]

    def compute(self, data):
        for index, augmentation in enumerate(self.augmentations):
            to_augment = getattr(augmentation, "keys_to_augment", [])
            if to_augment:
                for k, v in data.items():
                    if k in to_augment:
                        try:
                            data[k] = augmentation(v)
                        except (AttributeError, TypeError) as e:
                            error_message = (
                                f"Error applying augmentation '{augmentation.__class__.__name__}' to key '{k}' (value type: {type(v)}).\n"
                                "This may be due to using a transform that expects Tensor inputs with numpy arrays.\n"
                                "Please ensure you add a ToTensor augmentation (\"type\": \"ToTensor\") "
                                "at the appropriate position in the pipeline (often at the beginning if using torchvision transforms)."
                            )
                            raise type(e)(error_message) from e
                    else:
                        data[k] = v
            else:
                for k, v in data.items():
                    try:
                        data[k] = augmentation(v)
                    except (AttributeError, TypeError) as e:
                        error_message = (
                            f"Error applying augmentation '{augmentation.__class__.__name__}' to key '{k}' (value type: {type(v)}).\n"
                            "This may be due to using a transform that expects Tensor inputs with numpy arrays.\n"
                            "Please ensure you add a ToTensor augmentation (\"type\": \"ToTensor\") "
                            "at the appropriate position in the pipeline (often at the beginning if using torchvision transforms)."
                        )
                        raise type(e)(error_message) from e

        assert all([isinstance(v, (Tensor, List[Tensor])) for _, v in data.items()]), (
            f"End of augmentation pipeline must return Tensor(s). Got {[type(v) for v in data.values()]}.\n"
            'Add a ToTensor augmentation (\"type\": \"ToTensor\") at the end of the pipeline.'
        )

        return list(data.values())


def register_transforms() -> None:
    """
    Registers all valid transforms classes from torch.optim as subclasses of BaseDataAugmentation.
    """
    EXCLUDE_TRANSFORMS = ["compose", "Compose", "Lambda"]
    transforms_classes = inspect.getmembers(transforms, inspect.isclass)
    for name, cls in transforms_classes:
        if (
                name in EXCLUDE_TRANSFORMS
            or isinstance(cls, Enum)
            or issubclass(cls, Enum)
            or issubclass(cls, torch.Tensor)
        ):
            continue
        else:
            subclass = type(
                name,
                (BaseDataAugmentationWithKeys, cls),
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
        transform_class : The transform to generate the schema for.

    Returns:
        Dict[str, Schema]: A dictionary mapping parameter names to their corresponding schema.
    """
    config_schema = {}
    init_signature = inspect.signature(transform_class.__init__)
    for param_name, param in init_signature.parameters.items():
        if param_name in ["self", "params", "defaults"]:
            continue

        optional = param.default != inspect.Parameter.empty
        default = param.default if optional else None

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


class BaseDataAugmentationWithKeys(BaseDataAugmentation):
    """
    BaseDataAugmentationWithKeys is an abstract class that extends BaseDataAugmentation

    This class is used to define data augmentation techniques that require a list of keys.

    Configuration:
        - **keys_to_augment** (List[str]): List of keys in the dataset to apply augmentation (default: []). **If empty, the augmentation is applied to all keys.**
    """

    config_schema = {
        "keys_to_augment": Schema(List[str], default=[]),
    }
    pass


class NoiseDataAugmentation(BaseDataAugmentationWithKeys):
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


class ExtendedDataAugmentation(BaseDataAugmentationWithKeys):
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
