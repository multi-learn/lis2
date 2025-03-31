import random
from collections import defaultdict
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from configurable import Schema

from .dataset import BaseDataset


class Fake3DDataset(BaseDataset):
    """
    Dataset for creating fake 3D volumes by adding an additional dimension to 2D patches.

    **Configurations:**
        - **dataset_path** (:class:`Union[Path, str]`): Path to the HDF5 file containing the dataset.
        - **learning_mode** (:class:`str`): Mode of learning strategy. Default is "conservative".
        - **data_augmentation** (:class:`str`): Type of data augmentation applied. Default is "noise".
        - **normalization_mode** (:class:`str`): Normalization mode used for preprocessing. Default is "none".
        - **input_data_noise** (:class:`float`): Amount of noise added to the input data. Default is 0.
        - **output_data_noise** (:class:`float`): Amount of noise added to the output data. Default is 0.
        - **stride** (:class:`int`): Stride used for indexing patches. Default is 2.
        - **fold_assignments** (:class:`defaultdict`, optional): Dictionary mapping folds to dataset indices.
        - **fold_list** (:class:`list`, optional): List of folds to include in the dataset.
        - **toEncode** (:class:`list`, optional): List of parameters to be encoded. Default is [].
    """

    config_schema = {
        "dataset_path": Schema(Union[Path, str]),
        "learning_mode": Schema(str, default="conservative"),
        "data_augmentation": Schema(str, default="noise"),
        "normalization_mode": Schema(str, default="none"),
        "input_data_noise": Schema(float, default=0),
        "output_data_noise": Schema(float, default=0),
        "toEncode": Schema(list, optional=True, default=[]),
        "stride": Schema(int, default=2),
        "fold_assignments": Schema(defaultdict, optional=True),
        "fold_list": Schema(list, optional=True),
    }

    def __init__(self):
        """
        Initializes the Fake3DDataset class by loading data and setting parameters.
        """
        self.data = h5py.File(self.dataset_path, "r")
        self.patches = self.data["patches"]
        self.spines = self.data["spines"]
        self.labelled = self.data["labelled"]

        self.rng = random.Random()
        self.parameters_to_encode = set(self.toEncode)
        self.dic_mapping = self.create_mapping()

    def __len__(self):
        """
        Returns:
            int: Number of elements in the dataset.
        """
        return len(self.dic_mapping)

    def __getitem__(self, i):
        """
        Retrieves a sample from the dataset.

        Args:
            i (:class:`int`): Index of the sample.

        Returns:
            dict: Dictionary containing the sample with 3D patch, labelled data, and optionally spines.
        """
        idx = self.dic_mapping[i]
        patch = self.data["patches"][idx]
        labelled = self.data["labelled"][idx]

        parameters_to_encode_values = {}
        for param in self.parameters_to_encode:
            try:
                parameters_to_encode_values[param] = self.data[param][idx]
            except KeyError:
                self.logger.error(
                    f"Parameter {param} is not in the HDF5 file provided. Please check the configuration or the data."
                )
                raise ValueError

        spines = self.data["spines"][idx] if "spines" in self.data else None

        patch, spines, labelled = self.apply_data_augmentation(
            [patch, spines, labelled],
            self.data_augmentation,
            self.input_data_noise,
            self.output_data_noise,
            self.rng,
        )

        patch_3d = np.expand_dims(np.repeat(spines, 32, axis=2), axis=3)
        spines_3d = np.expand_dims(np.repeat(spines, 32, axis=2), axis=3) if spines is not None else None
        labelled_3d = np.expand_dims(np.repeat(spines, 32, axis=2), axis=3)

        return self._create_sample(patch_3d, spines_3d, labelled_3d, parameters_to_encode_values)

    def _create_sample(self, patch, spines, labelled, parameters_to_encode_values):
        """
        Creates a 3D sample from the data.

        Args:
            patch (:class:`np.ndarray`): The input 3D volume.
            spines (:class:`np.ndarray`, optional): The target/output 3D volume.
            labelled (:class:`np.ndarray`): The 3D volume indicating labelled pixels (1 for labelled, 0 otherwise).
            parameters_to_encode_values (:class:`dict`): Values of the parameters to encode.

        Returns:
            dict: Dictionary containing `patch`, `labelled`, and optionally `spines`.
        """
        patch = torch.from_numpy(patch).float().permute(3, 2, 0, 1)
        labelled = torch.from_numpy(labelled).float().permute(3, 2, 0, 1)

        sample = {"patch": patch, "labelled": labelled}

        if spines is not None:
            sample["target"] = torch.from_numpy(spines).float().permute(3, 2, 0, 1)

        for key, value in parameters_to_encode_values.items():
            sample[key] = torch.from_numpy(value)

        return sample

    def create_mapping(self):
        """
        Creates a mapping of dataset indices based on fold assignments.

        Returns:
            dict: Dictionary mapping indices from 0 to the total number of patches considering the stride.
        """
        if not self.fold_assignments and not self.fold_list:
            return {i: i for i in range(len(self.patches))}

        dic_mapping = {}
        i = 0
        for fold in self.fold_list:
            for idx in self.fold_assignments.get(fold, [])[:: self.stride]:
                dic_mapping[i] = idx
                i += 1

        if not dic_mapping:
            raise ValueError(f"No data found for the given fold assignments for {self.name}.")
        return dic_mapping
