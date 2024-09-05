from unittest import TestCase

import numpy as np

import deep_filaments.utils.normalizer as norma


class TestNormalizer(TestCase):
    def test_normalize_direct(self):
        t = np.ones((5, 5))
        t[1, 1] = 30
        t[0, 0] = -2
        res = norma.normalize_direct(t)
        self.assertEqual(res[1, 1], 1)
        self.assertEqual(res[0, 0], 0)

    def test_normalize_histo(self):
        t = np.ones((10, 10))
        t[1, 1] = 20
        t[0, 0] = -2
        res = norma.normalize_histo(t)
        self.assertEqual(res[1, 1], 1)
        self.assertEqual(res[0, 0], 0.01)

    def test_normalize_adapt_histo(self):
        t = np.ones((20, 20))
        t[1, 1] = 50
        t[0, 0] = -10
        res = norma.normalize_adapt_histo(t)
        self.assertEqual(res[1, 1], 1)
        self.assertTrue(res[0, 0] < 0.03)
