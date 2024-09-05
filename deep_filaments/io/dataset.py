"""Pytorch dataset of filaments."""
import random

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

import deep_filaments.utils.transformers as tf
import deep_filaments.utils.normalizer as norma


def create_sample(patch, positions, spines, missing, background, labelled, normed, missmap):
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
    positions = torch.from_numpy(positions)

    # permute data so the channels will be in the 0th axis (pytorch compability)
    patch = patch.permute(2, 0, 1)

    sample = {
        "patch": patch,
        "position": positions
    }

    if spines is not None:
        spines = torch.from_numpy(spines)
        spines = spines.permute(2, 0, 1)
        sample["target"] = spines

    if missing is not None:
        missing = torch.from_numpy(missing)
        missing = missing.permute(2, 0, 1)
        sample["missing"] = missing

    if background is not None:
        background = torch.from_numpy(background)
        background = background.permute(2, 0, 1)
        sample["background"] = background

    if labelled is not None:
        labelled = torch.from_numpy(labelled)
        labelled = labelled.permute(2, 0, 1)
        sample["labelled"] = labelled

    if normed is not None:
        normed = torch.from_numpy(normed)
        normed = normed.permute(2, 0, 1)
        sample["normed"] = normed

    if missmap is not None:
        missmap = torch.from_numpy(missmap)
        missmap = missmap.permute(2, 0, 1)
        sample["missmap"] = missmap

    return sample

def apply_data_augmentation(
    data, augmentation_style, input_noise_var, output_noise_var, random_gen
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

    if augmentation_style == 1:
        new_data = tf.apply_noise_transform(
            data, input_noise_var=input_noise_var, output_noise_var=output_noise_var
        )
    elif augmentation_style == 2:
        noise_list = [
            0,
        ] * len(data)
        noise_list[0:2] = [input_noise_var, output_noise_var]
        new_data = tf.apply_extended_transform(data, random_gen, noise_var=noise_list)

    for i in range(len(new_data)):
        new_data[i] = np.clip(np.array(new_data[i], dtype="f"), 0.0, 1.0)

    return new_data


class FilamentsDataset(Dataset):
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

    def __init__(
        self,
        dataset_path: str,
        learning_mode: str = "conservative",
        data_augmentation: int = 0,
        normalization_mode: str = "none",
        input_data_noise: float = 0,
        output_data_noise: float = 0,
        missmap: bool = False,
    ):
        data = h5py.File(dataset_path, "r")
        self.patches = data["patches"]
        self.positions = data["positions"]
        if "spines" in data:
            self.spines = data["spines"]
        else:
            self.spines = None
        if "missing" in data:
            self.missing = data["missing"]
        else:
            self.missing = None
        if "background" in data:
            self.background = data["background"]
        else:
            self.background = None
        if "normed" in data:
            self.normed = data["normed"]
        else:
            self.normed = None
        self.min = 4.160104636600882e+20
        self.max = 3.367595174607165e+23
        self.data_augmentation = data_augmentation
        self.rng = random.Random()
        if learning_mode == "conservative":
            self.learning_mode = 0
        elif learning_mode == "oneclass":
            self.learning_mode = 1
        elif learning_mode == "onevsall":
            self.learning_mode = 2
        self.normalize = True if normalization_mode != "none" else False
        self.input_data_noise = input_data_noise
        self.output_data_noise = output_data_noise
        self.missingmap = missmap
        self.normalization_mode = 0 if normalization_mode == "direct" else 1

        if "spines" in data and len(self.patches) != len(self.spines):
            raise ValueError(
                "Invalid dataset, the number of patches "
                f"{len(self.patches)} is not equal to the number of "
                f"spines {len(self.spines)}."
            )

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        patch = self.patches[idx]
        positions = self.positions[idx]
        if self.spines is not None:
            spines = self.spines[idx]
        else:
            spines = None
        if self.missing is not None:
            missing = self.missing[idx]
        else:
            missing = None
        if self.background is not None:
            background = self.background[idx]
        else:
            background = None
        if self.normed is not None:
            normed = self.normed[idx]
        else:
            normed = None
        
        if background is not None and spines is not  None:
            if self.learning_mode == 0:
                labelled = background + spines
                labelled[labelled > 0] = 1
            elif self.learning_mode == 1:
                labelled = spines.copy()
            elif self.learning_mode == 2:
                labelled = np.ones_like(patch)
        else:
            labelled = None

        if self.normalize:
            midx = patch > 0
            if midx.any():
                if self.normalization_mode == 0:
                    patch[midx] = norma.normalize_direct(patch[midx])
                elif self.normalization_mode == 1:
                    patch[midx] = np.log10(patch[midx])
                    patch[midx] = norma.normalize_direct(patch[midx], np.log10(self.min), np.log10(self.max))

        if self.data_augmentation > 0:
            patch, spines, missing, background, labelled, normed = apply_data_augmentation(
                [patch, spines, missing, background, labelled, normed],
                self.data_augmentation,
                self.input_data_noise,
                self.output_data_noise,
                self.rng,
            )
        
        if self.missingmap:
            missmap = missing_map(patch, 0)
        else:
            missmap = None

        sample = create_sample(patch, positions, spines, missing, background, labelled, normed, missmap)

        return sample

class OneDpixelDataset(Dataset):
    """
    Read filaments and their segmentation.

    Parameters
    ----------
    data: array of pixels
    labels: array of labels
    normalize: bool
        Apply inflight pixel normalization
    """

    def __init__(
        self,
        data,
        labels,
        normalize: bool = False,
        saturation: bool = False,
    ):
        self.data = data
        self.labels = labels
        self.normalize = normalize
        self.saturation = saturation
        self.alpha = 3 / np.max(data, axis=(0, 1))

    def __len__(self):
        """Return number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        data = self.data[idx]
        label = self.labels[idx]
        if self.normalize:
            midx = data != 0
            if midx.any():
                data = norma.normalize_direct(data)

        if self.saturation:
            data = np.tanh(self.alpha * data) / self.alpha
        sample = {
            "data": torch.from_numpy(data),
            "label": torch.tensor(label)
        }
        return sample

def missing_map(image, value):
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