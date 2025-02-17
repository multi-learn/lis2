import abc
import random
from pathlib import Path
from typing import Union, List, Dict, Any

import numpy as np
import torch
from configurable import TypedConfigurable, Schema


class DataAugmentations:
    def __init__(self, augmentations_configs: List[Dict[str, Any]]):
        """
        Initializes the Augmentations class with a list of augmentation configurations.

        Args:
            augmentations_configs (List[Dict[str, Any]]): List of augmentations configurations.
        """
        self.augmentations = [
            BaseDataAugmentation.from_config(config) for config in augmentations_configs
        ]

    def compute(self, data):
        for augmentation in self.augmentations:
            data = augmentation.apply_data_augmentation(data)
        return data


class BaseDataAugmentation(abc.ABC, TypedConfigurable):
    @abc.abstractmethod
    def apply_data_augmentation(self):
        pass

    @abc.abstractmethod
    def transformation(self):
        pass


class NoiseDataAugmentation(BaseDataAugmentation):
    config_schema = {
        "type": Schema(str),
        "input_noise_var": Schema(float, default=0),
        "output_noise_var": Schema(float, default=0),
    }

    def apply_data_augmentation(self, data):
        """
        Apply a data augmentation scheme to given data

        Parameters
        ----------
        data: list[np.ndarray]
            A list of patches (input, output, others...)
        augmentation_style: int
            The kind of transformation (1: only noise, 2: noise + flip + rotation)
            with flip+rotation.
        input_noise_var: float
            The noise variance on input data
        output_noise_var: float
            The noise variance on the output data

        Returns
        -------
        The list of transformed patches in the same order as input.
        """

        new_data = self.transformation(data)

        for i in range(len(new_data)):
            new_data[i] = np.clip(np.array(new_data[i], dtype="f"), 0.0, 1.0)

        return new_data

    def transformation(self, data):
        """
        Apply a random transform to a list of data (full_version)
        Note: only apply on the first item

        Parameters
        ----------
        data_list: list
            A list of data
        input_noise_var: float
            The variance of the additional noise (input)
        output_noise_var: float
            The variance of the additional noise (output)

        Returns
        -------
        A list of transform data
        """
        in_noise = np.array(
            np.random.standard_normal(data[0].shape) * self.input_noise_var, dtype="f"
        )
        out_noise = np.array(
            np.random.standard_normal(data[0].shape) * self.output_noise_var, dtype="f"
        )
        n_in_data = data[0] + in_noise
        n_out_data = data[1] + out_noise
        res = data.copy()
        res[0] = n_in_data
        res[1] = n_out_data
        return res


class ExtendedDataAugmentation(BaseDataAugmentation):
    config_schema = {
        "type": Schema(str),
        "input_noise_var": Schema(float, default=0),
        "output_noise_var": Schema(float, default=0),
        "random_gen": Schema(random.Random),
    }

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

        noise_list = [
            0,
        ] * len(data)
        noise_list[0:2] = [self.input_noise_var, self.output_noise_var]
        new_data = self.transformation(data_list=data, noise_var=noise_list)

        for i in range(len(new_data)):
            new_data[i] = np.clip(np.array(new_data[i], dtype="f"), 0.0, 1.0)

        return new_data

    def transformation(self, data_list, noise_list):
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
        res = []
        for data, variance in zip(data_list, noise_list):
            noise = np.array(
                np.random.standard_normal(data.shape) * variance, dtype="f"
            )
            data = data + noise
            res.append(self.random_augment(data, n_tf))
        return res

    def random_augment(data, num_tf):
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
