import unittest
from PNRIA.torch_c.dataset import FoldsController
from pathlib import Path

from PNRIA.tests.config.config import PATH_TO_SAMPLE_DATASET


class TestFoldsController(unittest.TestCase):

    def fold_controler_config(self):
        config_dict = {
            "train_ratio": 0.5,
            "dataset_path": PATH_TO_SAMPLE_DATASET + "patches.h5",
            "indices_path": PATH_TO_SAMPLE_DATASET + "indices.pkl",
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

    def test_generate_kfold_splits(self):
        config_dict = self.fold_controler_config()
        controller = FoldsController.from_config(config_dict)
        splits = controller.generate_kfold_splits(controller.k, controller.k_train)
        self.assertEqual(len(splits), 4)
        self.assertEqual(len(splits[0][0]), 2)

    def test_fold_assignments(self):
        config_dict = self.fold_controler_config()
        controller = FoldsController.from_config(config_dict)
        area_groups, fold_assignments = controller.create_folds_random_by_area(
            k=controller.k,
            area_size=controller.area_size,
            patch_size=controller.patch_size,
            overlap=controller.overlap,
        )
        assert Path(controller.indices_path).exists()
        assert len(area_groups) == 64
        assert len(fold_assignments) == 4

        max_len = max(len(area_groups[key]) for key in area_groups)
        min_len = min(len(area_groups[key]) for key in area_groups)

        assert max_len == 1089
        assert min_len == 90
