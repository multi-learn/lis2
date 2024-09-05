from unittest import TestCase

import numpy as np

import deep_filaments.utils.missing_data as missing


class TestMissingData(TestCase):
    def test_test_missing_data_value(self):
        t = np.ones((10,))
        self.assertTrue(missing.test_missing_data_value(t, 0))
        self.assertFalse(missing.test_missing_data_value(t, 11))
        t[0] = np.nan
        self.assertFalse(missing.test_missing_data_value(t, -1))

    def test_fill_missing_data(self):
        t = np.ones((10, 10))
        self.assertTrue((missing.fill_missing_data(t, 0) == np.ones((10, 10))).all())
        t[1, 1] = 2
        t[1, 2] = 4
        res = missing.fill_missing_data(t, 1.5)
        self.assertTrue(res[0, 0] == 3)
        self.assertTrue(res[1, 1] == 2)
        t[0, 0] = np.nan
        res = missing.fill_missing_data(t, 1.5)
        self.assertTrue(res[0, 0] == 3)
        self.assertTrue(res[1, 2] == 4)
