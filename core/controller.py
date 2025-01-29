import pickle
from collections import defaultdict
from pathlib import Path
from typing import Union

import h5py

from configs.config import Schema, Customizable


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
        "dataset_path": Schema(Union[Path, str]),
        "k": Schema(int, aliases=["nb_folds"], default=1),
        "k_train": Schema(float, aliases=["train_ratio"], default=0.80),
        "indices_path": Schema(Union[Path, str]),
        "save_indices": Schema(bool),
        "area_size": Schema(int, default=64),
        "patch_size": Schema(int, default=32),
        "overlap": Schema(int, default=0),
    }

    def __init__(self):
        self.dataset = h5py.File(self.dataset_path, "r")
        self.indices_path = self.indices_path
        self.splits = generate_kfold_splits(self.k, self.k_train)
        self.area_groups, self.fold_assignments = self._create_folds_random_by_area()

    def preconditions(self):
        # TODO : Better modularity, depending on k and k train
        assert (self.k * self.k_train) % 2 == 0, "train_ratio must be even"

    def _create_folds_random_by_area(self):
        """
        Distribute patches into k folds by assigning areas to folds in a round-robin manner.

        Returns:
            area_groups (dict): Dictionary mapping area coordinates to a list of patch indices.
            fold_assignments (dict): Dictionary mapping fold numbers to a list of patch indices.
        """
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

        return dict(area_groups), dict(fold_assignments)


# region Utils


def generate_kfold_splits(k, k_train):
    """
    Generate exactly k splits where each fold takes turns being the validation and test set.
    Handles special case for k = 1.

    Parameters:
        k (int): Total number of folds.
        k_train (int): Number of folds in the training set (for k > 1, k_train = k - 2).

    Returns:
        list of tuples: Each tuple contains (i_train, i_valid, i_test).

    Raises:
        ValueError: If invalid parameters are provided.
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

    elif k % 2 != 0:
        raise ValueError("k must be pair")

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
