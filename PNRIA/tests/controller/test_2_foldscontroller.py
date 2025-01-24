import unittest
import pytest
from pathlib import Path

from PNRIA.tests.config.config import PATH_TO_SAMPLE_DATASET
from PNRIA.torch_c.controller import FoldsController, generate_kfold_splits


class TestFoldsController(unittest.TestCase):

    def fold_controler_config(self):
        config_dict = {
            "train_ratio": 0.5,
            "dataset_path": PATH_TO_SAMPLE_DATASET / "patches.h5",
            "indices_path": PATH_TO_SAMPLE_DATASET / "indices.pkl",
            "save_indices": True,
            "nb_folds": 4,
            "area_size": 64,
            "patch_size": 32,
        }
        return config_dict

    def test_fold_controler_init(self):
        config_dict = self.fold_controler_config()
        controller = FoldsController.from_config(config_dict)
        self.assertEqual(
            controller.k_train, 0.5
        ), "ratio should be equal to configured value"
        self.assertEqual(controller.save_indices, True)
        self.assertEqual(controller.k, 4)

    def test_generate_splits(self):
        splits = generate_kfold_splits(10, 0.60)
        assert splits == [
            ([4, 5, 6, 7, 8, 9], [0, 1], [2, 3]),
            ([0, 5, 6, 7, 8, 9], [1, 2], [3, 4]),
            ([0, 1, 6, 7, 8, 9], [2, 3], [4, 5]),
            ([0, 1, 2, 7, 8, 9], [3, 4], [5, 6]),
            ([0, 1, 2, 3, 8, 9], [4, 5], [6, 7]),
            ([0, 1, 2, 3, 4, 9], [5, 6], [7, 8]),
            ([0, 1, 2, 3, 4, 5], [6, 7], [8, 9]),
            ([1, 2, 3, 4, 5, 6], [7, 8], [9, 0]),
            ([2, 3, 4, 5, 6, 7], [8, 9], [0, 1]),
            ([3, 4, 5, 6, 7, 8], [9, 0], [1, 2]),
        ]

        splits = generate_kfold_splits(1, 0.80)
        assert splits == [([0, 1, 2, 3, 4, 5, 6, 7], [8], [9])]
        with pytest.raises(ValueError):
            splits = generate_kfold_splits(1, 0.50)
        with pytest.raises(ValueError):
            splits = generate_kfold_splits(8, 0.60)

    def test_generate_kfold_splits(self):
        config_dict = self.fold_controler_config()
        controller = FoldsController.from_config(config_dict)
        splits = controller.splits
        self.assertEqual(len(splits), 4)
        self.assertEqual(len(splits[0][0]), 2)

    def test_fold_assignments(self):
        config_dict = self.fold_controler_config()
        controller = FoldsController.from_config(config_dict)
        area_groups = controller.area_groups
        fold_assignments = controller.fold_assignments
        assert Path(controller.indices_path).exists()
        assert len(area_groups) == 64
        assert len(fold_assignments) == 4

        max_len = max(len(area_groups[key]) for key in area_groups)
        min_len = min(len(area_groups[key]) for key in area_groups)

        assert max_len == 1089
        assert min_len == 90
