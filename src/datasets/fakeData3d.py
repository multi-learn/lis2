import random
from collections import defaultdict
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import torch
from configurable import Schema

from src.datasets.dataset import BaseDataset


class Fake3DDataset(BaseDataset):
    """
    Dataset for creating fake 3D volumes by adding an additional dimension to 2D patches.

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
        return len(self.dic_mapping)

    def __getitem__(self, i):
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

        patch_3d = np.expand_dims(np.repeat(spines, 32, axis=2), axis=3)
        spines_3d = (
            np.expand_dims(np.repeat(spines, 32, axis=2), axis=3)
            if spines is not None
            else None
        )
        labelled_3d = np.expand_dims(np.repeat(spines, 32, axis=2), axis=3)

        sample = self._create_sample(
            patch_3d, spines_3d, labelled_3d, parameters_to_encode_values
        )

        return sample

    def _create_sample(self, patch, spines, labelled, parameters_to_encode_values):
        """
        Create a 3D sample from the data.

        Parameters
        ----------
        patch: np.ndarray
            The input 3D volume.
        spines: np.ndarray
            The target/output 3D volume.
        labelled: np.ndarray
            The 3D volume indicating where the labelled pixel are (1 for labelled, 0 else).
        parameters_to_encode_values: dict
            The values of the parameters to encode.

        Returns
        -------
        A dictionary forming the samples in torch.Tensor format.
        """
        patch = torch.from_numpy(patch).float()
        labelled = torch.from_numpy(labelled).float()

        patch = patch.permute(3, 2, 0, 1)
        labelled = labelled.permute(3, 2, 0, 1)

        sample = {
            "patch": patch,
            "labelled": labelled,
        }

        if spines is not None:
            spines = torch.from_numpy(spines).float()
            spines = spines.permute(3, 2, 0, 1)
            sample["target"] = spines

        if parameters_to_encode_values:
            for key in parameters_to_encode_values:
                sample[key] = torch.from_numpy(parameters_to_encode_values[key])

        return sample

    def create_mapping(self):
        if not self.fold_assignments and not self.fold_list:
            self.logger.debug(
                "No fold assignments or fold list provided. Using all data."
            )
            return {i: i for i in range(len(self.patches))}
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
        return dic_mapping
