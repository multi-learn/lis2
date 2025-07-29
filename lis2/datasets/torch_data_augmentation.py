from numbers import Number
from typing import Dict, Sequence

import torch
from configurable import Schema
from torch import Tensor

from lis2.datasets.data_augmentation import BaseDataAugmentationWithKeys
from lis2.utils.distributed import get_rank


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "int32": torch.int32,
        "int64": torch.int64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128
    }
    dtype_str = dtype_str.lower()
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        raise ValueError(f"Invalid dtype string: '{dtype_str}'. Supported dtypes are: {', '.join(dtype_map.keys())}")


class ToTensor(BaseDataAugmentationWithKeys):
    """
    Converts the specified keys in the dataset to PyTorch tensors.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **dType** (str): Data type to convert the tensors to (default: "float32").
        - **force_device** (str): Force the tensors to be on a specific device (default: None).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "ToTensor"
            dType: "float32"
            force_device: "cuda"

    """

    config_schema = {"dType": Schema(str, default="float32"),
                     "force_device": Schema(str, optional=True)}

    def __init__(self):
        self.dType = str_to_dtype(self.dType)

    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Converts all input data to tensors and ensures that they are on the same device as the model.
        """
        return {key: torch.tensor(value, dtype=self.dType) for key, value in data.items()}


import torch
from torchvision.transforms.v2 import RandomRotation as TVRandomRotation
from typing import Dict, List, Union
from torch import Tensor


class RandomRotation(BaseDataAugmentationWithKeys):
    """
    Applies random rotation to the specified keys in the dataset.

    A single rotation angle is sampled and applied consistently across all specified keys.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique.
        - **keys_to_augment** (List[str]): List of keys in the dataset to apply the augmentation to (default: [] => all).
        - **degrees** (Union[float, Tuple[float, float]]): Range of degrees for rotation. If a float is given,
          the range is (-degrees, +degrees). Default: 30.
        - **expand** (bool): Whether to expand the image to fit the rotated result (default: False).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "RandomRotation"
            name: "RotationAugment"
            keys_to_augment: ["patch", "spines"]
            degrees: 45
            expand: True

    see also: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomRotation
    """

    config_schema = {
        "keys_to_augment": Schema(List[str], default=[]),
        "degrees": Schema(Union[Number, Sequence]),
        "expand": Schema(bool, default=False),
    }

    def __init__(self):
        self.transform = TVRandomRotation(degrees=self.degrees, expand=self.expand)

    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        angle = self.transform.get_params(self.degrees)
        return {key: self.transform(value, angle) for key, value in data.items()}


from random import random


class RandomHorizontalFlip(BaseDataAugmentationWithKeys):
    """
    Applies random horizontal flip to all keys in the dataset.

    A single flip decision is sampled and applied consistently across all keys.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique.
        - **p** (float): Probability of flipping the input horizontally (default: 0.5).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "RandomHorizontalFlip"
            name: "HorizontalFlipAugment"
            p: 0.5

    See also : https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip
    """

    config_schema = {
        "p": Schema(float, default=0.5),
    }

    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply horizontal flip with probability `p` to all keys in the dataset.

        Args:
            data (Dict[str, Tensor]): Input dataset to augment.

        Returns:
            Dict[str, Tensor]: Augmented dataset.
        """
        if random() < self.p:
            for key in data:
                data[key] = torch.flip(data[key], dims=[-1])
        return data


class RandomVerticalFlip(BaseDataAugmentationWithKeys):
    """
    Applies random vertical flip to all keys in the dataset.

    A single flip decision is sampled and applied consistently across all keys.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique.
        - **p** (float): Probability of flipping the input vertically (default: 0.5).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "RandomVerticalFlip"
            name: "VerticalFlipAugment"
            p: 0.5

    See also : https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomVerticalFlip
    """

    config_schema = {
        "p": Schema(float, default=0.5),
    }

    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply vertical flip with probability `p` to all keys in the dataset.

        Args:
            data (Dict[str, Tensor]): Input dataset to augment.

        Returns:
            Dict[str, Tensor]: Augmented dataset.
        """
        if random() < self.p:
            for key in data:
                data[key] = torch.flip(data[key], dims=[-2])
        return data

from torchvision.transforms.v2 import Normalize as TVNormalize

class Normalize(BaseDataAugmentationWithKeys):
    """
    Applies normalization to all keys in the dataset.

    Configuration:
        - **type** (str): Type of data augmentation (required).
        - **name** (str): Name of the augmentation technique.
        - **mean** (float): The expected mean (default: [0.5, 0.5, 0.5]).
        - **std** (float): The expected std (default: [0.5, 0.5, 0.5]).

    Example Configuration (YAML):
        .. code-block:: yaml

            type: "Normalize"
            name: "SimpleNormalize"
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]

    See also : https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.Normalize.html
    """

    config_schema = {
        "mean": Schema(Sequence[float], default=[0.5, 0.5, 0.5]),
        "std": Schema(Sequence[float], default=[0.5, 0.5, 0.5]),
    }

    def __init__(self):
        self.tf = TVNormalize(mean=self.mean, std=self.std)

    def transform(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {key: self.tf(value) for key, value in data.items()}
