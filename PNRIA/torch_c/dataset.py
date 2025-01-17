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
from PNRIA.configs.config import (
    TypedCustomizable,
    Schema,
    Customizable,
)


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

        if self.data_augmentation:
            assert self.data_augmentation in {
                "noise",
                "extended",
            }, "data_augmentation must be one of {'noise', 'extended'}"

        assert len(self.dic_mapping) != 0, "Dataset is empty"

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


class FoldsController(Customizable):
    """
    A class to manage and generate k-fold splits for dataset training, validation, and testing,
    with support for patch-level data organization and area-based grouping.

    Attributes:
        dataset_path (str): Path to the dataset file.
        k (int): Total number of folds for k-fold cross-validation.
        k_train (float): Ratio of folds to be used for training.
        indices_path (str): Path to store or load precomputed fold indices.
        save_indices (bool): Whether to save computed indices to a file.
        area_size (int): Size of areas to group patches for fold assignment.
        patch_size (int): Size of each patch in the dataset.
        overlap (int): Number of pixels overlapping between adjacent areas.

    Methods:
        generate_kfold_splits(k, k_train):
            Generate exactly k splits where each fold takes turns being the validation and test set.

        create_folds_random_by_area(k, area_size=64, patch_size=32, overlap=0):
            Distribute patches into k folds by grouping them into areas and assigning areas to folds
            in a round-robin manner.
    """

    config_schema = {
        "dataset_path": Schema(str),
        "k": Schema(int, aliases=["nb_folds"], default=1),
        "k_train": Schema(float, aliases=["train_ratio"], default=0.80),
        "indices_path": Schema(str),
        "save_indices": Schema(bool),
        "area_size": Schema(int, default=64),
        "patch_size": Schema(int, default=32),
        "overlap": Schema(int, default=0),
    }

    def __init__(self):
        # TODO : Better modularity, depending on k and k train
        assert (self.k * self.k_train) % 2 == 0, "train_ratio must be even"
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
        folds = list(range(0, k))
        if (k * k_train) != k - 2:
            raise ValueError("k = k_train must be k - 2.")

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

    def create_folds_random_by_area(self):
        """
        Distribute patches into k folds by assigning areas to folds in a round-robin manner.

        Parameters:
            k (int): Total number of folds.
            area_size (int): Size of the areas in which patches are grouped.
            patch_size (int): Size of each patch.
            overlap (int): Number of pixels overlapping between areas.

        Returns:
            area_groups (dict): Dictionary mapping area coordinates to a list of patch indices.
            fold_assignments (dict): Dictionary mapping fold numbers to a list of patch indices.
        """
        if self.indices_path.exists():
            self.logger.info(
                "Indice file already exists. Skipping indices computation and using the existing one"
            )
            # Load area_groups back
            with open(self.indices_path, "rb") as f:
                area_groups = pickle.load(f)

        else:
            patches = self.dataset["patches"]
            positions = self.dataset["positions"]
            len_patches = patches.shape[0]
            self.logger.info(
                f"No indices file found. Attributing indices to fold and storing result in {self.indices_path}"
            )
            # Group patches by area based on their positions
            area_groups = defaultdict(list)
            for idx in range(len_patches):
                y1 = positions[idx][0][0]
                x1 = positions[idx][1][0]

                # Calculate the top-left corner of the area this patch belongs to
                # Adjust logic based on overlap
                area_key = (
                    int((y1) // (self.area_size)),
                    int((x1) // (self.area_size)),
                )

                area_start_y = area_key[0] * self.area_size
                area_start_x = area_key[1] * self.area_size
                area_end_y = area_start_y + self.area_size
                area_end_x = area_start_x + self.area_size

                # Check if patch is inside the area, accounting for the overlap
                patch_end_y = y1 + self.patch_size
                patch_end_x = x1 + self.patch_size
                if (
                    patch_end_y > area_end_y + self.overlap
                    or patch_end_x > area_end_x + self.overlap
                ):
                    continue

                area_groups[area_key].append(idx)

            if self.save_indices:
                # Save area_groups to a file
                with open(self.indices_path, "wb") as f:
                    pickle.dump(area_groups, f)

        self.logger.info("Assigning area to folds")
        # Distribute areas to folds using round-robin
        fold_assignments = defaultdict(list)
        for fold_idx, area_key in enumerate(area_groups):
            fold = fold_idx % self.k
            fold_assignments[fold].extend(area_groups[area_key])

        return area_groups, fold_assignments
