import unittest

import numpy as np
import pytest
import torch

from src.datasets.data_augmentation import DataAugmentations


class TestFilamentsDataset(unittest.TestCase):

    def data_augmentation_config(self):
        config_data_augmentations = [
            {"type": "ToTensor"},
            {
                "type": "NoiseDataAugmentation",
                "name": "input",
                "keys_to_augment": ["patch"],
            },
            {
                "type": "NoiseDataAugmentation",
                "name": "output",
                "keys_to_augment": ["spines"],
            },
        ]
        return config_data_augmentations

    def test_np(self):
        config = self.data_augmentation_config()
        data_augmentations = DataAugmentations(augmentations_configs=config)
        data1 = np.random.rand(20, 20, 1)
        var = {"input": data1}
        output = data_augmentations.compute(var)
        self.assertEqual(output[0].shape, (1, 20, 20))
        data1 = np.random.rand(20, 20, 1)
        data2 = np.random.rand(20, 20, 1)
        var = {"input1": data1, "input2": data2}
        output1, output2 = data_augmentations.compute(var)
        self.assertEqual(output1.shape, (1, 20, 20))
        self.assertEqual(output2.shape, (1, 20, 20))

    def test_tensor(self):
        config = self.data_augmentation_config()
        data_augmentations = DataAugmentations(augmentations_configs=config)
        tensor = torch.randn(1, 3, 32, 32)
        var = {"input": tensor}
        with pytest.raises(TypeError):
            data_augmentations.compute(var)

    def test_wrong_configs(self):
        config = [
            {
                "type": "NoiseDataAugmentation",
                "name": "input",
                "keys_to_augment": ["input"],
            },
        ]

        with pytest.raises(TypeError):
            data_augmentations = DataAugmentations(augmentations_configs=config)
            data1 = np.random.rand(20, 20, 1)
            var = {"input": data1}
            data_augmentations.compute(var)

        config = [
            {
                "type": "NoiseDataAugmentation",
                "name": "input",
                "keys_to_augment": ["input"],
            },
            {"type": "ToTensor", "keys_to_augment": ["input"]},
        ]

        with pytest.raises(TypeError):
            data_augmentations = DataAugmentations(augmentations_configs=config)
            data1 = np.random.rand(20, 20, 1)
            var = {"input": data1}
            data_augmentations.compute(var)


def test_good_usage_with_tensor():
    config = [
        {
            "type": "ExtendedDataAugmentation",
            "name": "input",
            "keys_to_augment": ["input"],
        }, ]

    data_augmentations = DataAugmentations(augmentations_configs=config)
    tensor = torch.randn(3, 32, 32)
    var = {"input": tensor}
    output = data_augmentations.compute(var)
    assert output[0].shape == torch.Size([32, 3, 32])
