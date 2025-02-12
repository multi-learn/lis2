from pathlib import Path

import pytest

from src.controller import FoldsController, generate_kfold_splits
from src.preprocessing import BasePatchExtraction
from tests.config.config import PATH_TO_SAMPLE_DATASET, TempDir


class TestFoldsController(TempDir):

    # Not tested here. Needed to make this test self-sufficient
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

    def fold_controller_config(self):
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

    def test_1_fold_controller(self):

        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        preprocessor.extract_patches()

        config = self.fold_controller_config()
        controller = FoldsController.from_config(config)
        self.assertEqual(
            controller.k_train, 0.5
        ), "ratio should be equal to configured value"
        self.assertEqual(controller.save_indices, True)
        self.assertEqual(controller.k, 4)

    def test_2_generate_splits(self):
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

    def test_3_generate_kfold_splits(self):
        config = self.fold_controller_config()
        controller = FoldsController.from_config(config)

        splits = controller.splits
        self.assertEqual(len(splits), 4)
        self.assertEqual(len(splits[0][0]), 2)

    def test_4_fold_assignments(self):
        config_dict = self.fold_controller_config()
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
