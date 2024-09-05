from unittest import TestCase

import numpy as np

import deep_filaments.metrics.quality_measures as qm


class TestQMetrics(TestCase):
    def test_get_missed_structures(self):
        segmentation = np.zeros((10, 10))
        segmentation[0, 1] = 1
        segmentation[0, 2] = 1
        segmentation[2, 4] = 0.1

        groundt = np.zeros((10, 10))
        groundt[0, 1] = 1
        groundt[0, 2] = 1
        groundt[2, 4] = 1
        groundt[2, 5] = 1
        groundt[2, 6] = 1

        truth = np.zeros((10, 10))
        truth[2, 4] = 1
        truth[2, 5] = 1
        truth[2, 6] = 1

        res = qm.get_missed_structures(segmentation, groundt, 0.2)
        self.assertTrue(np.linalg.norm(res - truth) < 1e-5)

    def test_positive_part_accuracy(self):
        segmentation = np.zeros((10, 10))
        segmentation[0, 1] = 1
        segmentation[0, 2] = 1
        segmentation[2, 4] = 0.1

        groundt = np.zeros((10, 10))
        groundt[0, 1] = 1
        groundt[0, 2] = 1
        groundt[2, 4] = 1
        groundt[2, 5] = 1
        groundt[2, 6] = 1

        res = qm.positive_part_accuracy(segmentation, groundt)
        np.testing.assert_almost_equal(res, 0.42)

    def test_compute_missed_structures(self):
        segmentation = np.zeros((10, 10))
        segmentation[0, 1] = 1
        segmentation[0, 2] = 1
        segmentation[2, 4] = 0.1

        groundt = np.zeros((10, 10))
        groundt[0, 1] = 1
        groundt[0, 2] = 1
        groundt[2, 4] = 1
        groundt[2, 5] = 1
        groundt[2, 6] = 1

        truth = np.zeros((10, 10))
        truth[2, 4] = 1
        truth[2, 5] = 1
        truth[2, 6] = 1

        res = qm.compute_missed_structures(segmentation, groundt, 0.2)
        np.testing.assert_almost_equal(res, 0.5)
