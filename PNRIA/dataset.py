"""Pytorch dataset of filaments."""
import abc
from enum import Enum
import random

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

import deep_filaments.utils.transformers as tf
import deep_filaments.utils.normalizer as norma
from PNRIA.configs.config import TypedCustomizable, Schema

class BaseDataset(abc.ABC, TypedCustomizable, Dataset):

    @abc.abstractmethod
    def __len__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __getitem__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _create_sample(self, *args, **kwargs):
        pass

    def apply_data_augmentation(
        self, data, augmentation_style, input_noise_var, output_noise_var, random_gen
    ):
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
        random_gen: random.Random
            The random generator

        Returns
        -------
        The list of transformed patches in the same order as input.
        """
        new_data = data

        if augmentation_style == 'noise':
            new_data = tf.apply_noise_transform(
                data, input_noise_var=input_noise_var, output_noise_var=output_noise_var
            )
        elif augmentation_style == 'extended':
            noise_list = [
                0,
            ] * len(data)
            noise_list[0:2] = [input_noise_var, output_noise_var]
            new_data = tf.apply_extended_transform(data, random_gen, noise_var=noise_list)

        for i in range(len(new_data)):
            new_data[i] = np.clip(np.array(new_data[i], dtype="f"), 0.0, 1.0)

        return new_data

    def missing_map(self, image, value):
        """
        Detect if there is missing data in a given image.

        Parameters
        ----------
        image: np.ndarray
            An image.

        Returns
        -------
        res: np.ndarray
            The map of the missing data.
        """
        res = np.ones(image.shape)
        res[image <= value] = 0.0
        return res


class FilamentsDataset(BaseDataset):
    """
    Read filaments and their segmentation.

    Parameters
    ----------
    dataset_path : str
        The name of the hdf5 file with the data.
    data_augmentation : int
        Apply the given model of random transformation for data augmentation, default is 0.
    normalize: bool
        Apply inflight patch normalization
    input_data_noise : float
        The noise level for data augmentation on input data
    output_data_noise : float
        The noise level for data augmentation on output data
    """

    config_schema = {
        'dataset_path': Schema(str),
        'learning_mode': Schema(str, ["conservative"]),
        'data_augmentation': Schema(str, optional=True),
        'normalization_mode': Schema(str, ["none"]),
        'missingmap': Schema(bool, aliases=['missmap']),
        "input_data_noise": Schema(float, 0),
        "output_data_noise": Schema(float, 0),
        'min': Schema(float, default=4.160104636600882e+20),
        'max': Schema(float, default=3.367595174607165e+23),
        'toEncode': Schema(list),
    }

    def __init__(self):
        data = h5py.File(self.dataset_path, "r")
        parameters_to_encode = set()
        for item in self.toEncode:
            parameters_to_encode.add(item)
        self.parameters_to_encode = parameters_to_encode
        self.data = data

        self.rng = random.Random()
        assert self.learning_mode in {"conservative", "oneclass", "onevsall"}, "Learning_mode must be one of {conservative, oneclass, onevsall}"
        self.normalize = True if self.normalization_mode != "none" else False
        self.normalization_mode = 0 if self.normalization_mode == "direct" else 1

        if "spines" in data and len(data['patches']) != len(data['spines']):
            raise ValueError(
                "Invalid dataset, the number of patches "
                f"{len(self.patches)} is not equal to the number of "
                f"spines {len(self.spines)}."
            )

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.data['patches'])

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        
        patch = self.data['patches'][idx]
        spines = self.data['spines'][idx]
        labelled = self.data['labelled'][idx]

        # For optional parameters such as position
        parameters_to_encode_values = {}

        for param in self.parameters_to_encode:
            try:
                parameters_to_encode_values[param] = self.data[param][idx]
            except:
                print(f"Parameter {param} is not in the hdf5 file provided. Please check the configuration or the data")


        if 'spines' in self.data and self.data['spines'] is not None:
            spines = self.data['spines'][idx]
        else:
            spines = None

        if self.data_augmentation:
            assert self.data_augmentation in {'noise', 'extended'}, "Learning_mode must be one of {'noise', 'extended'}"
            patch, spines, labelled = self.apply_data_augmentation(
                [patch, spines, labelled],
                self.data_augmentation,
                self.input_data_noise,
                self.output_data_noise,
                self.rng,
            )

        sample = self._create_sample(patch, spines, labelled, parameters_to_encode_values)
        
        return sample

    def _create_sample(self, patch, spines, labelled, parameters_to_encode_values):
        """
        Create a sample from the data

        Parameters
        ----------
        patch: np.ndarray
            The input data patch
        target: np.ndarray
            The target/output data patch
        missing: np.ndarray
            The patch indicating the missing data (0 for missing, 1 else)
        background: np.ndarray
            The patch indicating the position of the background pixels (1 for background, 0 else)
        labelled: np.ndarray
            The patch indicating where the labelled pixel are (1 for labelled, 0 else).

        Returns
        -------
        A dictionary forming the samples in torch.Tensor format
        """
        patch = torch.from_numpy(patch)
        spines = torch.from_numpy(spines)
        labelled = torch.from_numpy(labelled)

        # permute data so the channels will be in the 0th axis (pytorch compability)
        patch = patch.permute(2, 0, 1)
        spines = spines.permute(2, 0, 1)
        labelled = labelled.permute(2, 0, 1)


        sample = {
            "patch": patch,
            "target": spines,
            "labelled": labelled
        }
        
        for key in parameters_to_encode_values:
            sample[key] = torch.from_numpy(parameters_to_encode_values[key])

        return sample