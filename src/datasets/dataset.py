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

class FilamentsDataset(BaseDataset):
    """
    FilamentsDataset for loading and managing preprocessed data stored in an HDF5 file.

    This dataset class loads a dataset from an HDF5 file, extracts relevant patches,
    encodes specific parameters, and applies optional data augmentation techniques.

    Configuration:

    - **dataset_path** (Union[Path, str]): Path to the HDF5 file containing the dataset.
    - **learning_mode** (str): The dataset's learning mode (default: 'conservative').
    - **data_augmentations** (List[Config]): Type of data augmentation to apply (default: None).
    - **toEncode** (List[str]): List of parameters to encode (default: []).
    - **stride** (int): Stride value for mapping patch indices (default: 1).
    - **fold_assignments** (Optional[Dict]): Dictionary defining fold assignments (default: None).
    - **fold_list** (Optional[List]): List of folds to use in training (default: None).
    - **use_all_patches** (bool): Whether to include all patches in the dataset (default: False).

    Example Configuration (YAML):
        .. code-block:: yaml

            dataset_path: "/path/to/dataset.h5"
            learning_mode: "conservative"
            data_augmentation: [{"type": "NoiseDataAugmentation"}]
            toEncode: ["param1", "param2"]
            stride: 2
            fold_assignments: {"fold1": [0, 2, 4], "fold2": [1, 3, 5]}
            fold_list: [[[1, 2], [3], [4]], [[3, 4], [1], [2]]]
            use_all_patches: True
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
            The target data.
        labelled: np.ndarray
            Value indicating where the labelled pixels are (1 for labelled, 0 else).
        parameters_to_encode_values: dict
            The values of the parameters to encode. (for example position)

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
        """
        Maps dataset indices based on fold assignments and stride.
        If no fold assignments or fold list provided, uses all patches.
        If fold assigments provided, If NOT use all patches, ensures
        that at least one pixel is annotated in a patch for each fold.
        """
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
