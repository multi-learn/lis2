from unittest import TestCase

import deep_filaments.io.utils as utils


class TestUtils(TestCase):
    def test_get_sorted_file_list(self):
        lst = utils.get_sorted_file_list("data")
        self.assertEqual(lst[0], "nh2C_mosaic349356.fits.gz")

    def test_get_mosaic_angles(self):
        angles = utils.get_mosaic_angles("data/nh2C_mosaic349356.fits.gz")
        self.assertEqual(angles, "349356")
