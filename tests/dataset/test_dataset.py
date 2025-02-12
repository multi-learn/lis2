import torch

from src.controller import FoldsController
from src.preprocessing import BasePatchExtraction
from src.datasets.dataset import BaseDataset
from tests.config.config import PATH_TO_SAMPLE_DATASET, TempDir


class TestFilamentsDataset(TempDir):

    def filament_dataset_config(self):
        config_dict = {
            "type": "FilamentsDataset",
            "dataset_path": self.temp_dir / "patches.h5",
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
    # Fold controller test should be run before dataset test
    def controller_config(self):
        config_dict = {
            "train_ratio": 0.5,
            "dataset_path": self.temp_dir / "patches.h5",
            "indices_path": self.temp_dir / "indices.pkl",
            "save_indices": True,
            "nb_folds": 4,
            "area_size": 64,
            "patch_size": 32,
        }

        return config_dict

    def preprocessing_config(self):
        config = {
            "type": "PatchExtraction",
            "image": PATH_TO_SAMPLE_DATASET / "sample_image.fits",
            "target": PATH_TO_SAMPLE_DATASET / "sample_target.fits",
            "missing": PATH_TO_SAMPLE_DATASET / "sample_missing.fits",
            "background": PATH_TO_SAMPLE_DATASET / "sample_background.fits",
            "output": self.temp_dir,
            "patch_size": 32,
        }
        return config

    def test_dataset(self):
        dataset_config = self.filament_dataset_config()
        # Necessary to test dataset

        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        preprocessor.extract_patches()

        controller_config = self.controller_config()
        controller = FoldsController.from_config(controller_config)

        splits = controller.splits

        fold_assignments = controller.fold_assignments

        dataset_config["fold_assignments"] = fold_assignments
        dataset_config["fold_list"] = splits[0][0]

        dataset = BaseDataset.from_config(dataset_config)
        self.assertEqual(dataset.learning_mode, "conservative")
        self.assertEqual(dataset.data_augmentation, "noise")
        self.assertEqual(dataset.normalization_mode, "test")
        self.assertEqual(dataset.input_data_noise, 0.0)
        self.assertEqual(dataset.output_data_noise, 0.0)
        self.assertEqual(dataset.toEncode, ["positions"])
        self.assertEqual(dataset.stride, 3)
        self.assertEqual(dataset.fold_assignments, fold_assignments)
        self.assertEqual(dataset.fold_list, splits[0][0])

        assert len(dataset) == 8095
        assert len(dataset[0]) == 4
        assert list(dataset[0].keys()) == ["patch", "target", "labelled", "positions"]
        assert dataset[0]["patch"].shape == torch.Size([1, 32, 32])
        assert dataset[0]["labelled"].shape == torch.Size([1, 32, 32])
        assert dataset[0]["target"].shape == torch.Size([1, 32, 32])
        assert dataset[0]["positions"].shape == torch.Size([2, 2, 1])

    def test_dataset_use_all_patches(self):
        dataset_config = self.filament_dataset_config()
        # Necessary to test dataset

        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        preprocessor.extract_patches()

        controller_config = self.controller_config()
        controller = FoldsController.from_config(controller_config)

        splits = controller.splits

        fold_assignments = controller.fold_assignments

        dataset_config["fold_assignments"] = fold_assignments
        dataset_config["fold_list"] = splits[0][0]
        dataset_config["use_all_patches"] = True
        dataset = BaseDataset.from_config(dataset_config)
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
