import abc
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List

import h5py
from configurable import Schema, TypedConfigurable
from tqdm import tqdm


class FoldsController(TypedConfigurable):
    """
    Abstract base class for fold control.

    Defines the interface for classes that perform fold assignments using different strategies.
    """

    @abc.abstractmethod
    def _create_folds(self):
        pass

    def _create_areas(self):

        if type(self.indices_path) == str:
            self.indices_path = Path(self.indices_path)
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
            for idx in tqdm(range(len_patches)):
                y1 = positions[idx][0][0]
                x1 = positions[idx][1][0]

                # Calculate the top-left corner of the area this patch belongs to
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
        return area_groups


class RandomController(FoldsController):
    """
    RandomController for managing and generating k-fold splits with patch-level data organization
    and area-based grouping in a round-robin manner.

    This controller organizes dataset patches into areas and assigns these areas to different folds
    in a round-robin manner. It supports saving and loading precomputed fold indices for reproducibility
    and efficiency.

    Configuration:
        - **dataset_path** (Union[Path, str]): Path to the dataset file.
        - **k** (int): Total number of folds for k-fold cross-validation (default: 1).
        - **k_train** (float): Ratio of folds to be used for training (default: 0.80).
        - **indices_path** (Union[Path, str]): Path to store or load precomputed fold indices.
        - **save_indices** (bool): Whether to save computed indices to a file.
        - **area_size** (int): Size of areas to group patches for fold assignment (default: 64).
        - **patch_size** (int): Size of each patch in the dataset (default: 32).
        - **overlap** (int): Number of pixels overlapping between adjacent areas (default: 0).

    Example Configuration (YAML):
        .. code-block:: yaml

            dataset_path: "/path/to/dataset.h5"
            k: 5
            k_train: 0.80
            indices_path: "/path/to/indices.json"
            save_indices: True
            area_size: 64
            patch_size: 32
            overlap: 0
    """

    config_schema = {
        "dataset_path": Schema(Path),
        "k": Schema(int, aliases=["nb_folds"], default=1),
        "k_train": Schema(float, aliases=["train_ratio"], default=0.80),
        "indices_path": Schema(Path),
        "save_indices": Schema(bool),
        "area_size": Schema(int, default=64),
        "patch_size": Schema(int, default=32),
        "overlap": Schema(int, default=0),
    }

    def __init__(self):
        self.dataset = h5py.File(self.dataset_path, "r")
        self.indices_path = Path(
            self.indices_path
            / f"indices_{self.area_size}_{self.patch_size}_{self.overlap}.pkl"
        )
        self.splits = generate_kfold_splits(self.k, self.k_train)
        self.area_groups, self.fold_assignments = self._create_folds()

    def preconditions(self):
        assert self.overlap < self.area_size, "Overlap must be less than area size"

    def _create_folds(
        self,
    ) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
        """
        Distribute patches into k folds by assigning areas to folds in a round-robin manner.

        This method groups patches into areas based on their positions and assigns these areas to folds.
        If an indices file already exists, it loads the precomputed area groups from the file. Otherwise,
        it computes the area groups and optionally saves them to a file.

        Returns:
            Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]: A tuple containing two dictionaries:
                - area_groups (Dict[Tuple[int, int], List[int]]): Dictionary mapping area coordinates to a list of patch indices.
                - fold_assignments (Dict[int, List[int]]): Dictionary mapping fold numbers to a list of patch indices.
        """
        area_groups = self._create_areas()

        if self.k == 1:
            # If k==1, we still divide into 10 folds to distribute folds according to k_train
            nb_folds = 10
        else:
            nb_folds = self.k

        self.logger.info("Assigning area to folds")
        # Distribute areas to folds using round-robin
        fold_assignments = defaultdict(list)
        for fold_idx, area_key in enumerate(area_groups):
            fold = fold_idx % nb_folds
            fold_assignments[fold].extend(area_groups[area_key])

        return dict(area_groups), dict(fold_assignments)


class NaiveController(FoldsController):
    """
    Manage and generate k-fold splits. It creates the folds by grouping patches into areas
    and assigning areas to folds in a naive manner (i.e. divides the image into k equal-sized areas).

    This controller divides the dataset into k equal-sized areas and assigns them to different folds
    in a naive manner. It supports saving and loading precomputed fold indices for reproducibility.

    Configuration:
        - **dataset_path** (Union[Path, str]): Path to the dataset file.
        - **k** (int): Total number of folds for k-fold cross-validation (default: 1).
        - **k_train** (float): Ratio of folds to be used for training (default: 0.80).
        - **indices_path** (Union[Path, str]): Path to store or load precomputed fold indices.
        - **save_indices** (bool): Whether to save computed indices to a file.
        - **area_size** (int): Size of areas to group patches for fold assignment (default: 64).
        - **patch_size** (int): Size of each patch in the dataset (default: 32).
        - **overlap** (int): Number of pixels overlapping between adjacent areas (default: 0).

    Example Configuration (YAML):
        .. code-block:: yaml

            dataset_path: "/path/to/dataset.h5"
            k: 5
            k_train: 0.80
            indices_path: "/path/to/indices.json"
            save_indices: True
            area_size: 64
            patch_size: 32
            overlap: 0
    """

    config_schema = {
        "dataset_path": Schema(Path),
        "k": Schema(int, aliases=["nb_folds"], default=1),
        "k_train": Schema(float, aliases=["train_ratio"], default=0.80),
        "indices_path": Schema(Path),
        "save_indices": Schema(bool),
        "area_size": Schema(int, default=64),
        "patch_size": Schema(int, default=32),
        "overlap": Schema(int, default=0),
    }

    def __init__(self):
        self.dataset = h5py.File(self.dataset_path, "r")
        self.indices_path = Path(
            self.indices_path
            / f"indices_{self.area_size=}_{self.patch_size=}_{self.overlap=}.pkl"
        )
        self.splits = generate_kfold_splits(self.k, self.k_train)
        self.area_groups, self.fold_assignments = self._create_folds()

    def preconditions(self):
        assert self.overlap < self.area_size, "Overlap must be less than area size"

    def _create_folds(
        self,
    ) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
        """
        Distribute patches into k folds by splitting the image into k equal parts and assigning areas accordingly.

        This method groups patches into areas based on their positions and assigns these areas to folds.
        If an indices file already exists, it loads the precomputed area groups from the file. Otherwise,
        it computes the area groups and optionally saves them to a file.

        Returns:
            Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]: A tuple containing two dictionaries:
                - area_groups (Dict[Tuple[int, int], List[int]]): Dictionary mapping area coordinates to a list of patch indices.
                - fold_assignments (Dict[int, List[int]]): Dictionary mapping fold numbers to a list of patch indices.
        """

        area_groups = self._create_areas()

        self.logger.info("Assigning area to folds")
        # Distribute areas to folds by splitting the image into k equal parts
        fold_assignments = defaultdict(list)
        area_keys = list(area_groups.keys())
        num_areas = len(area_keys)

        if self.k == 1:
            # If k==1, we still divide into 10 folds to distribute folds according to k_train
            nb_folds = 10
        else:
            nb_folds = self.k

        areas_per_fold = num_areas // nb_folds

        for fold_idx in range(nb_folds):
            start_idx = fold_idx * areas_per_fold
            end_idx = (
                (fold_idx + 1) * areas_per_fold
                if fold_idx != nb_folds - 1
                else num_areas
            )
            for area_key in area_keys[start_idx:end_idx]:
                fold_assignments[fold_idx].extend(area_groups[area_key])

        return dict(area_groups), dict(fold_assignments)


# region Utils


def generate_kfold_splits(
    k: int, k_train: float
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Generate exactly k splits where each fold takes turns being the validation and test set.

    This method generates k-fold splits for cross-validation, ensuring that each fold is used once as the validation set
    and once as the test set. It handles the special case where k = 1 by returning a single split with all data in the
    training set and empty validation and test sets.

    Args:
        k (int): Total number of folds.
        k_train (int): Number of folds in the training set. For k > 1, k_train is typically k - 2.

    Returns:
        List[Tuple[List[int], List[int], List[int]]]: A list of tuples, where each tuple contains:
            - i_train (List[int]): Indices of the training set.
            - i_valid (List[int]): Indices of the validation set.
            - i_test (List[int]): Indices of the test set.

    Raises:
        ValueError: If invalid parameters are provided, such as k <= 0 or k_train >= k.
    """
    if k == 1:
        if not (0 <= k_train <= 1 and (10 * k_train) % 2 == 0):
            raise ValueError(
                "For k = 1, k_train must be between 0 and 1 and 10 * k_train must be even."
            )

        # Generate the groups
        total_elements = list(range(10))
        num_train = int(10 * k_train)
        num_valid_test = (10 - num_train) // 2

        i_train = total_elements[:num_train]
        i_valid = total_elements[num_train : num_train + num_valid_test]
        i_test = total_elements[num_train + num_valid_test :]

        return [(i_train, i_valid, i_test)]

    else:
        if k - (k * k_train) == 2:
            num_train = int(k * k_train)
            num_valid_test = (k - num_train) // 2

            if num_train + 2 * num_valid_test != k:
                raise ValueError(
                    f"Invalid split: The sum of train, valid, and test elements must equal to {k}."
                )

            splits = []
            folds = list(range(k))  # Folds from 0 to 9

            for i in range(k):
                # Compute validation and test sets
                i_valid = [folds[i]]
                i_test = [folds[(i + 1) % k]]

                # Training set: all elements not in validation or test
                i_train = [fold for fold in folds if fold not in i_valid + i_test]

                # Adjust training to match the desired proportion
                i_train = i_train[:num_train]

                splits.append((i_train, i_valid, i_test))

        elif k - (k * k_train) == 4:
            num_train = int(k * k_train)
            num_valid_test = (k - num_train) // 2

            if num_train + 2 * num_valid_test != k:
                raise ValueError(
                    f"Invalid split: The sum of train, valid, and test elements must equal to {k}."
                )

            splits = []
            folds = list(range(k))  # Folds from 0 to 9

            for i in range(k):
                # Compute validation and test sets
                i_valid = [folds[i], folds[(i + 1) % k]]
                i_test = [folds[(i + 2) % k], folds[(i + 3) % k]]

                # Training set: all elements not in validation or test
                i_train = [fold for fold in folds if fold not in i_valid + i_test]

                # Adjust training to match the desired proportion
                i_train = i_train[:num_train]

                splits.append((i_train, i_valid, i_test))

        else:
            raise ValueError("k - (k*k_train) must be equal to 2 or 4")

    return splits


# endregion
