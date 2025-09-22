from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from configurable import Schema, Config

from lis2.datasets.data_augmentation import DataAugmentations
from lis2.datasets.dataset import BaseDataset


class FilamentsDataset(BaseDataset):
    """
    FilamentsDataset for loading and managing preprocessed data stored in an HDF5 file.

    This dataset class loads a dataset from an HDF5 file, extracts relevant patches,
    encodes specific parameters, and applies optional data augmentation techniques. This dataset
    class is specifically designed to handle the k-folds methods provided in the Controller file.

    Configuration:
        - **dataset_path** (Union[Path, str]): Path to the HDF5 file containing the dataset.
        - **data_augmentations** (List[Config]): Type of data augmentation to apply (default: None).
        - **toEncode** (List[str]): List of parameters to encode (default: []).
        - **stride** (int): Stride value for mapping patch indices (default: 1).
        - **fold_assignments** (Optional[Dict]): Dictionary defining fold assignments (default: None).
        - **fold_list** (Optional[List]): List of folds to use in training (default: None).
        - **use_all_patches** (bool): Whether to include all patches in the dataset (default: False).

    Example Configuration (YAML):
        .. code-block:: yaml

            dataset_path: "/path/to/dataset.h5"
            data_augmentation: [{"type": "ToTensor"},
            {"type": "NoiseDataAugmentation", "name": "input", "to_augment": [0, 1], "keys_to_augment":["patch"]}]
            toEncode: ["param1", "param2"]
            stride: 2
            fold_assignments: {"fold1": [0, 2, 4], "fold2": [1, 3, 5]}
            fold_list: [[[1, 2], [3], [4]], [[3, 4], [1], [2]]]
            use_all_patches: True
    """

    config_schema = {
        "dataset_path": Schema(Path),
        "data_augmentations": Schema(
            List[Config], optional=True, default=[{"type": "ToTensor"}]
        ),
        "normalization": Schema(str, optional=True, default=None),
        "toEncode": Schema(list, optional=True, default=[]),
        "stride": Schema(int, default=1),
        "fold_assignments": Schema(dict, optional=True),
        "fold_list": Schema(list, optional=True),
        "use_all_patches": Schema(bool, default=False),
    }

    def __init__(self):
        self.data = None
        self.is_open = False

        parameters_to_encode = set()
        for item in self.toEncode:
            parameters_to_encode.add(item)
        self.parameters_to_encode = parameters_to_encode
        self.data_augmentations = DataAugmentations(self.data_augmentations)

    def _open_h5_file(self):
        self.data = h5py.File(self.dataset_path, "r")
        self.patches = self.data["patches"]
        self.spines = self.data["spines"]
        self.labelled = self.data["labelled"]
        if not hasattr(self, "dic_mapping"):
            self.create_mapping()
            self.len = len(self.dic_mapping)

    def __getstate__(self):
        if hasattr(super(), "__getstate__"):
            super().__getstate__()
        state = self.__dict__.copy()
        if "data" in state and state["data"] is not None:
            state["data"].close()
        for key in ("data", "patches", "spines", "labelled"):
            state.pop(key, None)
        state["dataset_path"] = self.dataset_path
        self.is_open = False
        return state

    def __setstate__(self, state):
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)
        self.__dict__.update(state)
        if "dataset_path" in state:
            self._open_h5_file()
            self.is_open = True

    def __len__(self):
        """Return number of samples in the dataset."""
        if not hasattr(self, "len"):
            self._open_h5_file()
            self.is_open = True
        return self.len

    def __getitem__(self, i):
        """Return a sample from the dataset."""
        if not self.is_open:
            self._open_h5_file()
            self.is_open = True
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

        if self.normalization:
            midx = patch > 0
            if midx.any():
                if self.normalization == "direct":
                    patch[midx] = normalize_direct(patch[midx])
                elif self.normalization == "log":
                    patch[midx] = np.log10(patch[midx])
                    patch[midx] = normalize_direct(
                        patch[midx], np.log10(self.min), np.log10(self.max)
                    )

        data_to_augment = {"patch": patch, "spines": spines, "labelled": labelled}
        data_augment = self.data_augmentations.compute(data_to_augment)
        sample = self._create_sample(
            **data_augment, parameters_to_encode_values=parameters_to_encode_values
        )
        return sample

    def preconditions(self):
        assert (self.fold_assignments is None and self.fold_list is None) or (
            self.fold_assignments is not None and self.fold_list is not None
        ), "fold_assignments and fold_list must either both be None or both defined."
        if self.normalization:
            assert (
                self.normalization == "direct" or self.normalization == "log"
            ), "normalization must be 'direct' or 'log'"

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


def normalize_direct(x, xmin=None, zmax=None):
    """
    Normalize the data by directly projecting them on [0, 1]

    Parameters
    ----------
    x: np.ndarray
        The data

    Returns
    -------
    A normalized copy of the data
    """
    z = x.copy()
    if xmin is None:
        xmin = np.min(x.flat)
    z -= xmin
    if zmax is None:
        zmax = np.max(z.flat)
    if zmax > 0:
        z /= zmax
    return z
