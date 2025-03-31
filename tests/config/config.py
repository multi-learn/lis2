import shutil
import tempfile
import unittest
from pathlib import Path

import configs

# Put the sample dataset in a folder, and write the path to the folder here
# PATH_TO_SAMPLE_DATASET = Path("/Configure/Your/Path")
PATH_PROJECT = Path(configs.__file__).parent.parent
PATH_TO_SAMPLE_DATASET = Path(PATH_PROJECT / "sample_merged/")


class TempDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a shared temporary directory
        cls.temp_dir = Path(tempfile.mkdtemp(dir=PATH_TO_SAMPLE_DATASET))
        print(f"Temporary directory created at: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        # Clean up the shared temporary directory
        shutil.rmtree(cls.temp_dir)
        print(f"Temporary directory removed: {cls.temp_dir}")
