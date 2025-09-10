import abc
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from configurable import TypedConfigurable, Schema, Config
from torch.utils.data import Dataset

from lis2.datasets.data_augmentation import DataAugmentations


class BaseDataset(TypedConfigurable, Dataset, metaclass=abc.ABCMeta):
    """
    BaseDataset for creating datasets.

    This abstract base class provides methods for detecting missing data.
    It is designed to be subclassed for specific dataset implementations.
    """

    @abc.abstractmethod
    def __len__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __getitem__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _create_sample(self, *args, **kwargs):
        """
        Creates a data sample for the dataset.

        Args:
            args: Additional arguments for sample creation.
            kwargs: Additional keyword arguments for sample creation.

        Returns:
            Any: The created sample data.
        """
        pass

    def missing_map(self, image, value):
        """
        Detects missing data in a given image.

        This method generates a binary mask indicating missing data in an image.
        Pixels with values less than or equal to the specified threshold are marked as missing.

        Args:
            image (np.ndarray): The input image to analyze.
            value (float): The threshold below which data is considered missing.

        Returns:
            np.ndarray: A binary mask indicating missing data (1 for valid data, 0 for missing).
        """
        res = np.ones(image.shape)
        res[image <= value] = 0.0
        return res
