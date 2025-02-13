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
    """
    BaseDataset for creating a dataset with data augmentation and missing data detection.

    BaseDataset is an abstract base class that provides methods for data augmentation and
    missing data detection. It is intended to be subclassed to create specific datasets.

    Configuration:

    name (str): The name of the dataset.
    augmentation_style (str): The style of data augmentation to apply.
        Default is 'noise'.
    input_noise_var (float): The variance of noise to add to input data.
        Default is 0.01.
    output_noise_var (float): The variance of noise to add to output data.
        Default is 0.01.
    random_gen (random.Random): The random number generator to use for augmentation.
        Default is random.Random().

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "example_dataset"
            augmentation_style: "extended"
            input_noise_var: 0.02
            output_noise_var: 0.02
            random_gen: random.Random(42)

    Aliases:

    Dataset
    AugmentedDataset
    """

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
        value: float
            The threshold value below which data is considered missing.

        Returns
        -------
        np.ndarray
            The map of the missing data.
        """
        res = np.ones(image.shape)
        res[image <= value] = 0.0
        return res


class FilamentsDataset(BaseDataset):
    """
    FilamentsDataset for handling preprocessed data stored in an HDF5 file.

    This class loads a dataset from a specified HDF5 file path, extracts patches,
    spines, and labelled data, and initializes a random number generator. It also
    encodes parameters specified in `toEncode` and maps patch indices to a continuous
    range based on defined folds and stride.

    Configuration:

    name (str): The name of the dataset.
    dataset_path (Union[Path, str]): The path to the HDF5 file containing the dataset.
    learning_mode (str): The learning mode for the dataset. Default is 'conservative'.
    data_augmentation (str, optional): The type of data augmentation to apply.
        Default is None.
    normalization_mode (str): The normalization mode for the dataset. Default is 'none'.
    input_data_noise (float): The noise variance on input data. Default is 0.
    output_data_noise (float): The noise variance on output data. Default is 0.
    toEncode (list, optional): A list of parameters to encode. Default is [].
    stride (int): The stride for mapping patch indices. Default is 1.
    fold_assignments (dict, optional): A dictionary of fold assignments. Default is None.
    fold_list (list, optional): A list of folds to use. Default is None.
    use_all_patches (bool): Whether to use all patches. Default is False.

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "filaments_dataset"
            dataset_path: "/path/to/dataset.h5"
            learning_mode: "conservative"
            data_augmentation: "extended"
            normalization_mode: "none"
            input_data_noise: 0.01
            output_data_noise: 0.01
            toEncode: ["param1", "param2"]
            stride: 2
            fold_assignments: {"fold1": [0, 1, 2], "fold2": [3, 4, 5]}
            fold_list: ["fold1", "fold2"]
            use_all_patches: True

    Aliases:

    Dataset
    FilamentsData
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

        spines = (
            self.data["spines"][idx]
            if "spines" in self.data and self.data["spines"] is not None
            else None
        )

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
            The input data patch.
        spines: np.ndarray
            The target/output data patch.
        labelled: np.ndarray
            The patch indicating where the labelled pixels are (1 for labelled, 0 else).
        parameters_to_encode_values: dict
            The values of the parameters to encode.

        Returns
        -------
        dict
            A dictionary forming the samples in torch.Tensor format.
        """
        patch = torch.from_numpy(patch)
        spines = torch.from_numpy(spines)
        labelled = torch.from_numpy(labelled)

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
