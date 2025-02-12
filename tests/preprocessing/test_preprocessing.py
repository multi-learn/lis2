import unittest
from pathlib import Path

import h5py

from src.preprocessing import BasePatchExtraction
from tests.config.config import PATH_TO_SAMPLE_DATASET, TempDir

unittest.TestLoader.sortTestMethodsUsing = None


class TestPreprocessing(TempDir):

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

    def test_a_preprocessing_init(self):
        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        assert preprocessor.image.shape == (500, 500)
        assert preprocessor.target.shape == (500, 500)
        assert preprocessor.missing.shape == (500, 500)
        assert preprocessor.background.shape == (500, 500)

    def test_b_extract_patches(self):
        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        preprocessor.extract_patches()

        patches_path = Path(self.temp_dir / "patches.h5")
        assert patches_path.exists()

        data = h5py.File(patches_path, "r")
        assert list(data.keys()) == ["labelled", "patches", "positions", "spines"]
        assert data["patches"].shape == (200844, 32, 32, 1)
        assert data["labelled"].shape == (200844, 32, 32, 1)
        assert data["spines"].shape == (200844, 32, 32, 1)
        assert data["positions"].shape == (200844, 2, 2, 1)
