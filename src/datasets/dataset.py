"""Pytorch dataset of filaments."""

import abc
import random
from pathlib import Path
from typing import Union, List

import h5py
import numpy as np
import torch
from configurable import TypedConfigurable, Schema, Config
from torch.utils.data import Dataset

from src.datasets.data_augmentation import DataAugmentations


class BaseDataset(abc.ABC, TypedConfigurable, Dataset):

    @abc.abstractmethod
    def __len__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __getitem__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _create_sample(self, *args, **kwargs):
        pass

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
    patches, spines, and labelled data. It applies data augmentations according
    to the configuration.
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
    parameters_to_encode : set
        A set of parameters to be encoded from `toEncode`.
    dic_mapping : dict
        A dictionary mapping indices from 0 to the total number of patches in the folds
        considering the stride.
    """

    config_schema = {
        "dataset_path": Schema(Union[Path, str]),
        "learning_mode": Schema(str, default="conservative"),
        "data_augmentations": Schema(List[Config], optional=True),
        "toEncode": Schema(list, optional=True, default=[]),
        "stride": Schema(int, default=1),
        "fold_assignments": Schema(dict, optional=True),
        "fold_list": Schema(list, optional=True),
        "use_all_patches": Schema(bool, default=False),
    }

    def __init__(self):

        self.data = h5py.File(self.dataset_path, "r")
        self.patches = self.data["patches"]
        self.spines = self.data["spines"]
        self.labelled = self.data["labelled"]

        parameters_to_encode = set()
        for item in self.toEncode:
            parameters_to_encode.add(item)
        self.parameters_to_encode = parameters_to_encode
        self.create_mapping()
        self.data_augmentations = DataAugmentations(self.data_augmentations)

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
        assert (self.fold_assignments is None and self.fold_list is None) or (
            self.fold_assignments is not None and self.fold_list is not None
        ), "fold_assignments and fold_list must either both be None or both defined."

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

        assert len(set(sample["target"].flatten())) > 1, "target is empty"

        if parameters_to_encode_values:
            for key in parameters_to_encode_values:
                sample[key] = torch.from_numpy(parameters_to_encode_values[key])

        return sample

    def create_mapping(self):
        if not self.fold_assignments and not self.fold_list:
            self.logger.debug(
                "No fold assignments or fold list provided. Using all data."
            )
            self.dic_mapping = {i: i for i in range(len(self.patches))}
        else:
            dic_mapping = {}
            i = 0
            for fold in self.fold_list:
                for idx in self.fold_assignments.get(fold, [])[:: self.stride]:
                    dic_mapping[i] = idx
                    i += 1
            if not dic_mapping:
                raise ValueError(
                    f"No data found for the given fold assignments for {self.name}."
                )
            self.dic_mapping = dic_mapping
            if not self.use_all_patches:
                dic_labelled = {}
                i = 0
                for k, v in dic_mapping.items():
                    if len(set(self.spines[v].flatten())) > 1:
                        dic_labelled[i] = v
                        i += 1
                self.dic_mapping_save = self.dic_mapping
                self.dic_mapping = dic_labelled
