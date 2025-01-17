import unittest
import torch
from pathlib import Path
import numpy as np
from PNRIA.torch_c.dataset import BaseDataset, FoldsController

from PNRIA.tests.config.config import PATH_TO_SAMPLE_DATASET


class TestFilamentsDataset(unittest.TestCase):

    def filament_dataset_config(self):
        config_dict = {
            "type": "FilamentsDataset",
            "dataset_path": PATH_TO_SAMPLE_DATASET / "patches.h5",
            "learning_mode": "conservative",
            "data_augmentation": "noise",
            "normalization_mode": "test",
            "input_data_noise": 0.0,
            "output_data_noise": 0.0,
            "toEncode": ["positions"],
            "stride": 3,
        }
        return config_dict

    # Needed for the test. Not tested Here.
    # Fold controler test should be run before dataset test
    def controller_config(self):
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

    def test_dataset(self):
        config_dict = self.filament_dataset_config()
        # Necessary to test dataset
        path_indices = Path(PATH_TO_SAMPLE_DATASET / "indices.pkl")
        assert (
            path_indices.exists()
        ), f"Indices file should exist, run test_foldcontroller.py. Stored at: {path_indices}"

        config_dict_controller = self.controller_config()
        controller = FoldsController.from_config(config_dict_controller)

        splits = FoldsController.generate_kfold_splits(controller.k, controller.k_train)

        area_groups, fold_assignments = controller.create_folds_random_by_area()

        config_dict["fold_assignments"] = fold_assignments
        config_dict["fold_list"] = splits[0][0]

        dataset = BaseDataset.from_config(config_dict)
        self.assertEqual(dataset.learning_mode, "conservative")
        self.assertEqual(dataset.data_augmentation, "noise")
        self.assertEqual(dataset.normalization_mode, "test")
        self.assertEqual(dataset.input_data_noise, 0.0)
        self.assertEqual(dataset.output_data_noise, 0.0)
        self.assertEqual(dataset.toEncode, ["positions"])
        self.assertEqual(dataset.stride, 3)
        self.assertEqual(dataset.fold_assignments, fold_assignments)
        self.assertEqual(dataset.fold_list, splits[0][0])

        assert len(dataset) == 9132
        assert len(dataset[0]) == 4
        assert list(dataset[0].keys()) == ["patch", "target", "labelled", "positions"]
        assert dataset[0]["patch"].shape == torch.Size([1, 32, 32])
        assert dataset[0]["labelled"].shape == torch.Size([1, 32, 32])
        assert dataset[0]["target"].shape == torch.Size([1, 32, 32])
        assert dataset[0]["positions"].shape == torch.Size([2, 2, 1])
