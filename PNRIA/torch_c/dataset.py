"""Pytorch dataset of filaments."""

import abc
import random
from collections import defaultdict
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import deep_filaments.utils.transformers as tf
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

        if augmentation_style == "noise":
            new_data = tf.apply_noise_transform(
                data, input_noise_var=input_noise_var, output_noise_var=output_noise_var
            )
        elif augmentation_style == "extended":
            noise_list = [
                0,
            ] * len(data)
            noise_list[0:2] = [input_noise_var, output_noise_var]
            new_data = tf.apply_extended_transform(
                data, random_gen, noise_var=noise_list
            )
        else:
            raise ValueError("data_augmentation must be one of {'noise', 'extended'}")

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
    Main dataset class. Used to handle the preprocessed data, stored in a HDF5 file.

    Loads the dataset from the specified HDF5 file path, extracts
    patches, spines, and labelled data, and initializes a random number generator.
    It also encodes parameters specified in `toEncode` and maps patch indices
    to a continuous range based on the defined folds and stride.

    Attributes
    ----------
    data : h5py.File
        The HDF5 file containing the dataset.
    patches : h5py.Dataset
        The dataset containing patch data.
    spines : h5py.Dataset
        The dataset containing spine data.
    labelled : h5py.Dataset
        The dataset containing labelled data.
    rng : random.Random
        A random number generator instance.
    parameters_to_encode : set
        A set of parameters to be encoded from `toEncode`.
    dic_mapping : dict
        A dictionary mapping indices from 0 to the total number of patches in the folds
        considering the stride.
    """

    config_schema = {
        "dataset_path": Schema(Union[Path, str]),
        "learning_mode": Schema(str, default="conservative"),
        "data_augmentation": Schema(str, optional=True),
        "normalization_mode": Schema(str, default="none"),
        "input_data_noise": Schema(float, default=0),
        "output_data_noise": Schema(float, default=0),
        "toEncode": Schema(list, optional=True, default=[]),
        "stride": Schema(int, default=1),
        "fold_assignments": Schema(defaultdict, optional=True),
        "fold_list": Schema(list, optional=True),
    }

    def __init__(self):

        self.data = h5py.File(self.dataset_path, "r")
        self.patches = self.data["patches"]
        self.spines = self.data["spines"]
        self.labelled = self.data["labelled"]

        self.rng = random.Random()

        parameters_to_encode = set()
        for item in self.toEncode:
            parameters_to_encode.add(item)
        self.parameters_to_encode = parameters_to_encode
        self.dic_mapping = self.create_mapping()

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.dic_mapping)

    def __getitem__(self, i):
        """Return a sample from the dataset."""

        # Must map the given id to the actual id number, according to the folds
        idx = self.dic_mapping[i]

        patch = self.data["patches"][idx]
        labelled = self.data["labelled"][idx]

        parameters_to_encode_values = {}

        for param in self.parameters_to_encode:
            try:
                parameters_to_encode_values[param] = self.data[param][idx]
            except:
                self.logger.error(
                    f"Parameter {param} is not in the hdf5 file provided. Please check the configuration or the data"
                )
                raise ValueError

        if "spines" in self.data and self.data["spines"] is not None:
            spines = self.data["spines"][idx]
        else:
            spines = None

            patch, spines, labelled = self.apply_data_augmentation(
                [patch, spines, labelled],
                self.data_augmentation,
                self.input_data_noise,
                self.output_data_noise,
                self.rng,
            )

        sample = self._create_sample(
            patch, spines, labelled, parameters_to_encode_values
        )

        return sample

    def preconditions(self):
        assert (self.fold_assignments is None and self.fold_list is None) or \
               (self.fold_assignments is not None and self.fold_list is not None), \
            "fold_assignments and fold_list must either both be None or both defined."
        if self.data_augmentation:
            assert self.data_augmentation in {
                "noise",
                "extended",
            }, "data_augmentation must be one of {'noise', 'extended'}"

    def _create_sample(self, patch, spines, labelled, parameters_to_encode_values):
        """
        Create a sample from the data.

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
        parameters_to_encode_values: dict
            The values of the parameters to encode.

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
            "labelled": labelled,
        }

        if parameters_to_encode_values:
            for key in parameters_to_encode_values:
                sample[key] = torch.from_numpy(parameters_to_encode_values[key])

        return sample

    def create_mapping(self):
        if not self.fold_assignments and not self.fold_list:
            self.logger.debug("No fold assignments or fold list provided. Using all data.")
            return {i: i for i in range(len(self.patches))}
        dic_mapping = {}
        i = 0
        for fold in self.fold_list:
            for idx in self.fold_assignments.get(fold, [])[::self.stride]:
                dic_mapping[i] = idx
                i += 1
        if not dic_mapping:
            raise ValueError(f"No data found for the given fold assignments for {self.name}.")
        return dic_mapping
