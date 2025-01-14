import unittest
from PNRIA.utils.preprocessing import BasePreprocessing
from pathlib import Path
import h5py

from PNRIA.tests.config.config import PATH_TO_SAMPLE_DATASET


class TestPreprocessing(unittest.TestCase):
    def preprocessing_config(self):
        config = {
            "type": "PatchExtraction",
            "image": PATH_TO_SAMPLE_DATASET + "sample_image.fits",
            "target": PATH_TO_SAMPLE_DATASET + "sample_target.fits",
            "missing": PATH_TO_SAMPLE_DATASET + "sample_missing.fits",
            "background": PATH_TO_SAMPLE_DATASET + "sample_background.fits",
            "output": PATH_TO_SAMPLE_DATASET,
            "patch_size": 32,
        }
        return config

    def test_preprocessing_init(self):
        config_dict = self.preprocessing_config()
        preprocessor = BasePreprocessing.from_config(config_dict)
        assert preprocessor.image.shape == (500, 500)
        assert preprocessor.target.shape == (500, 500)
        assert preprocessor.missing.shape == (500, 500)
        assert preprocessor.background.shape == (500, 500)

    def test_extract_patches(self):
        config_dict = self.preprocessing_config()
        preprocessor = BasePreprocessing.from_config(config_dict)
        preprocessor.extract_patches()

        patches_path = Path(PATH_TO_SAMPLE_DATASET + "patches.h5")
        assert patches_path.exists()

        data = h5py.File(patches_path, "r")
        assert list(data.keys()) == ["labelled", "patches", "positions", "spines"]
        assert data["patches"].shape == (200844, 32, 32, 1)
        assert data["labelled"].shape == (200844, 32, 32, 1)
        assert data["spines"].shape == (200844, 32, 32, 1)
        assert data["positions"].shape == (200844, 2, 2, 1)
