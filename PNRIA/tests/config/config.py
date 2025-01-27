from pathlib import Path
import unittest
import tempfile
import shutil
import deep_filaments

# Put the sample dataset in a folder, and write the path to the folder here
# PATH_TO_SAMPLE_DATASET = Path("/Configure/Your/Path")
PATH_PROJECT = Path(deep_filaments.__file__).parent.parent
PATH_TO_SAMPLE_DATASET = Path(PATH_PROJECT / "PNRIA/sample_merged/")


class TempDir(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a shared temporary directory
        cls.temp_dir = Path(tempfile.mkdtemp())
        print(f"Temporary directory created at: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        # Clean up the shared temporary directory
        shutil.rmtree(cls.temp_dir)
        print(f"Temporary directory removed: {cls.temp_dir}")
