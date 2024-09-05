from unittest import TestCase

import numpy as np
import skimage.draw as draw

import deep_filaments.extraction.distance as dist
import deep_filaments.extraction.skeleton as ske


class TestSkeleton(TestCase):
    def test_compute_skeleton(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        start = (2, 2)
        extent = (7, 7)
        rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
        img[rr, cc] = 1

        distance = dist.distance_map(img)
        res = ske.compute_skeleton(img, distance, -0.04)

        truth = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.assertTrue(np.linalg.norm(truth - res) < 1e-9)
