import shutil
import tempfile
import time
import unittest
from pathlib import Path

import configs

# Put the sample dataset in a folder, and write the path to the folder here
# PATH_TO_SAMPLE_DATASET = Path("/Configure/Your/Path")
PATH_PROJECT = Path(configs.__file__).parent.parent
PATH_TO_SAMPLE_DATASET = Path(PATH_PROJECT / "sample_merged/")


# This is from https://github.com/python/cpython/issues/128076
# Needed because of some weird issues with rmtree
def retrying_rmtree(d):
    for _ in range(5):
        try:
            return shutil.rmtree(d)
        except OSError as e:
            if e.strerror == "Directory not empty":
                # wait a bit and try again up to 5 tries
                time.sleep(0.01)
            else:
                raise
    raise RuntimeError(f"shutil.rmtree('{d}') failed with ENOTEMPTY five times")


class TempDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a shared temporary directory
        cls.temp_dir = Path(tempfile.mkdtemp(dir=PATH_TO_SAMPLE_DATASET))
        print(f"Temporary directory created at: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        # Clean up the shared temporary directory
        retrying_rmtree(cls.temp_dir)
        print(f"Temporary directory removed: {cls.temp_dir}")
