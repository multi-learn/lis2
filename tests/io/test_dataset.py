from unittest import TestCase

import numpy as np

from deep_filaments.io.dataset import FilamentsDataset, FilamentsDatasetV2


class TestDataset(TestCase):
    def test_FilamentsDataset(self):
        fd = FilamentsDataset("data/data_v1.h5")
        self.assertEqual(len(fd), 2)
        self.assertEqual(fd.data_augmentation, 0)
        self.assertFalse(fd.normalize)

        res = fd.__getitem__(0)
        self.assertEqual(len(res), 5)
        self.assertTrue(np.linalg.norm(res["patch"] - np.ones((1, 32, 32))) < 1e-5)
        self.assertTrue(np.linalg.norm(res["target"] - np.ones((1, 32, 32))) < 1e-5)
        self.assertTrue(np.linalg.norm(res["missing"] - np.ones((1, 32, 32))) < 1e-5)
        self.assertTrue(np.linalg.norm(res["background"] - np.ones((1, 32, 32))) < 1e-5)
        self.assertTrue(np.linalg.norm(res["labelled"] - np.ones((1, 32, 32))) < 1e-5)

    def test_FilamentsDatasetV2(self):
        fd = FilamentsDatasetV2("data/data_v2.h5", "train")
        self.assertEqual(len(fd), 2)
        self.assertEqual(fd.data_augmentation, 0)
        self.assertFalse(fd.normalize)

        res = fd.__getitem__(0)
        self.assertEqual(len(res), 5)
        self.assertTrue(np.linalg.norm(res["patch"] - np.ones((1, 32, 32))) < 1e-5)

        tg = np.zeros((1, 32, 32))
        tg[0, 20, 20] = 1
        self.assertTrue(np.linalg.norm(res["target"] - tg) < 1e-5)
        self.assertTrue(np.linalg.norm(res["missing"] - tg) < 1e-5)
        self.assertTrue(
            np.linalg.norm(res["background"] - np.zeros((1, 32, 32))) < 1e-5
        )
