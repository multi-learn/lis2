"""Pytorch dataset of filaments."""

import abc
import random
from pathlib import Path
from collections import defaultdict
import time
import pickle


import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

import deep_filaments.utils.transformers as tf
from PNRIA.configs.config import TypedCustomizable, Schema, Customizable


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
        "dataset_path": Schema(str),
        "learning_mode": Schema(str, default="conservative"),
        "data_augmentation": Schema(str, optional=True, default=None),
        "normalization_mode": Schema(str, optional=True, default=None),
        "input_data_noise": Schema(float, default=0),
        "output_data_noise": Schema(float, default=0),
        "toEncode": Schema(list, default=[]),
    }

    def __init__(self):
        data = h5py.File(self.dataset_path, "r")
        parameters_to_encode = set()
        for item in self.toEncode:
            parameters_to_encode.add(item)
        self.parameters_to_encode = parameters_to_encode
        self.data = data

        self.rng = random.Random()
        assert self.learning_mode in {
            "conservative",
            "oneclass",
            "onevsall",
        }, "Learning_mode must be one of {conservative, oneclass, onevsall}"
        self.normalize = True if self.normalization_mode != "none" else False
        self.normalization_mode = 0 if self.normalization_mode == "direct" else 1

        if "spines" in data and len(data["patches"]) != len(data["spines"]):
            raise ValueError(
                "Invalid dataset, the number of patches "
                f"{len(self.patches)} is not equal to the number of "
                f"spines {len(self.spines)}."
            )

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.data["patches"])

    def __getitem__(self, idx):
        """Return a sample from the dataset."""

        patch = self.data["patches"][idx]
        spines = self.data["spines"][idx]
        labelled = self.data["labelled"][idx]

        # For optional parameters such as position
        parameters_to_encode_values = {}

        for param in self.parameters_to_encode:
            try:
                parameters_to_encode_values[param] = self.data[param][idx]
            except:
                print(
                    f"Parameter {param} is not in the hdf5 file provided. Please check the configuration or the data"
                )

        if "spines" in self.data and self.data["spines"] is not None:
            spines = self.data["spines"][idx]
        else:
            spines = None

        if self.data_augmentation:
            assert self.data_augmentation in {
                "noise",
                "extended",
            }, "data_augmentation must be one of {'noise', 'extended'}"
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

        sample = {"patch": patch, "target": spines, "labelled": labelled}

        for key in parameters_to_encode_values:
            sample[key] = torch.from_numpy(parameters_to_encode_values[key])

        return sample


class KFoldsController(Customizable):
    config_schema = {
        "dataset_path": Schema(str),
        "k": Schema(int),
        "k_train": Schema(int),
        "indices_path": Schema(str),
        "save_indices": Schema(bool),
    }

    def __init__(self):
        # TODO : Better modularity, depending on k and k train
        """
        Initialize a KFoldsController.

        Parameters
        ----------
        self : KFoldsController
            The object to be initialized.

        Notes
        -----
        The parameter k_train must satisfy k_train = k - 2, where k is the total number of folds.
        """
        assert (self.k - self.k_train) % 2 == 0, "k - k_train must be even"
        self.dataset = h5py.File(self.dataset_path, "r")
        self.indices_path = Path(self.indices_path)

    def generate_kfold_splits(self, k, k_train):
        """
        Generate exactly k splits where each fold takes turns being the validation and test set.

        Parameters:
            k (int): Total number of folds.
            k_train (int): Number of folds in the training set (k_train = k - 2).

        Returns:
            list of tuples: Each tuple contains (i_train, i_valid, i_test).
        """
        folds = list(range(1, k + 1))
        if k_train != k - 2:
            raise ValueError("k_train must be k - 2.")

        splits = []

        for i in range(k):
            # Validation fold
            i_valid = [folds[i]]
            # Test fold (next in cyclic order)
            i_test = [folds[(i + 1) % k]]
            # Training folds are all other folds
            i_train = [fold for fold in folds if fold not in i_valid + i_test]

            splits.append((i_train, i_valid, i_test))

        return splits

    def create_folds_random_by_area(self, k, area_size=64):
        """
        Distribute patches into k folds by assigning areas to folds in a round-robin manner.

        Parameters:
            k (int): Total number of folds.
            area_size (int): Size of the areas in which patches are grouped.

        Returns:
            area_groups (dict): Dictionary mapping area coordinates to a list of patch indices.
            fold_assignments (dict): Dictionary mapping fold numbers to a list of patch indices.
        """
        if self.indices_path.exists():
            self.log.info(
                "Indice file already exists. Skipping indices computation and using the existing one"
            )
            # Load area_groups back
            area_groups = {}
            with open(self.indices_path, "rb") as f:
                area_groups = pickle.load(f)

        else:
            patches = self.dataset["patches"]
            positions = self.dataset["positions"]
            start = time.time()
            len_patches = patches.shape[0]
            self.log.info(
                f"No indices file found. Attributing indices to fold and storing result in {self.indices_path}"
            )
            # Group patches by area based on their positions
            area_groups = defaultdict(list)
            for idx in range(len_patches):
                y1 = positions[idx][0][0]
                x1 = positions[idx][1][0]
                # Calculate the top-left corner of the area this patch belongs to
                area_key = (
                    int(y1 // area_size),
                    int(x1 // area_size),
                )  # Convert to standard integers
                area_groups[area_key].append(idx)

            if self.save_indices:
                # Save area_groups to a file
                with open(self.indices_path, "wb") as f:
                    pickle.dump(area_groups, f)

        self.log.info("Assigning area to folds")
        # Distribute areas to folds using round-robin
        fold_assignments = defaultdict(list)
        for fold_idx, area_key in enumerate(area_groups):
            fold = fold_idx % k
            fold_assignments[fold].extend(area_groups[area_key])

        return area_groups, fold_assignments


class KFoldsFilamentsDataset(BaseDataset):

    config_schema = {
        "dataset_path": Schema(str),
        "learning_mode": Schema(str, default=["conservative"]),
        "data_augmentation": Schema(str, optional=True),
        "normalization_mode": Schema(str, default=["none"]),
        "input_data_noise": Schema(float, default=0),
        "output_data_noise": Schema(float, default=0),
        "toEncode": Schema(list),
        "stride": Schema(int, default=1),
        "fold_assignments": Schema(defaultdict),
        "fold_list": Schema(list),
    }

    def __init__(self):
        """
        Initialize the KFoldsFilamentsDataset.

        This initializer loads the dataset from the specified HDF5 file path, extracts
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

        self.data = h5py.File(self.dataset_path, "r")
        self.patches = self.data["patches"]
        self.spines = self.data["spines"]
        self.labelled = self.data["labelled"]

        self.rng = random.Random()

        parameters_to_encode = set()
        for item in self.toEncode:
            parameters_to_encode.add(item)
        self.parameters_to_encode = parameters_to_encode

        # Maps id from 0 to len total of patches in folds
        i = 0
        self.dic_mapping = dict()
        for fold in self.fold_list:
            for idx in self.fold_assignments[fold][:: self.stride]:
                self.dic_mapping[i] = idx
                i += 1

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.dic_mapping)

    def __getitem__(self, i):
        """Return a sample from the dataset."""

        # Must map the given id to the actual id number, according to the folds
        idx = self.dic_mapping[i]

        patch = self.data["patches"][idx]
        spines = self.data["spines"][idx]
        labelled = self.data["labelled"][idx]

        # For optional parameters such as position
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

        if self.data_augmentation:
            assert self.data_augmentation in {
                "noise",
                "extended",
            }, "data_augmentation must be one of {'noise', 'extended'}"
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
