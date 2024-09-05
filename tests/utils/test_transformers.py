import random
from unittest import TestCase

import numpy as np

import deep_filaments.utils.transformers as tf


class TestTransformers(TestCase):
    def test_apply_extended_transform(self):
        img = np.ones((1, 32, 32))
        res = tf.apply_extended_transform([img, img], random.Random(), [0.1, 0.1])
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 32, 32))

    def test_apply_noise_transform(self):
        img = np.ones((1, 32, 32))
        res = tf.apply_noise_transform([img, img], 0.01, 0.01)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].shape, (1, 32, 32))
