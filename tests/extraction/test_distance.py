from unittest import TestCase

import numpy as np
import skimage.draw as draw

import deep_filaments.extraction.distance as dist


class TestDistance(TestCase):
    def test_distance_map(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        start = (2, 2)
        extent = (7, 7)
        rr, cc = draw.rectangle(start, extent=extent, shape=img.shape)
        img[rr, cc] = 1

        result = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        distances = dist.distance_map(img)
        self.assertTrue(np.linalg.norm(result - distances) < 1e-8)
