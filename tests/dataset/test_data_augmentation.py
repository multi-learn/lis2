import numpy as np
import pytest
import torch

from lis2.datasets.data_augmentation import DataAugmentations


@pytest.fixture
def data_augmentation_config():
    """Fixture for the data augmentation configuration."""
    return [
        {"type": "ToTensor", "force_device": "cpu"},
        {
            "type": "NoiseDataAugmentation",
            "name": "input",
            "keys_to_augment": ["input"],
        },
        {
            "type": "NoiseDataAugmentation",
            "name": "output",
            "keys_to_augment": ["input"],
        },
    ]


def test_data_augmentation_np(data_augmentation_config):
    """Test data augmentation on numpy arrays."""
    data_augmentations = DataAugmentations(augmentations_configs=data_augmentation_config)

    data1 = np.random.rand(1, 20, 20)
    var = {"input": data1}
    output = data_augmentations.compute(var)
    assert output["input"].shape == (1, 20, 20), "The 'input' output shape should be (1, 20, 20)"

    data2 = np.random.rand(1, 20, 20)
    var = {"input1": data1, "input2": data2}
    with pytest.raises(KeyError):
        data_augmentations.compute(var)


def test_data_augmentation_tensor(data_augmentation_config):
    """Test data augmentation on PyTorch tensors."""
    data_augmentations = DataAugmentations(augmentations_configs=data_augmentation_config)

    tensor = torch.randn(1, 3, 32, 32)
    var = {"input": tensor}
    output = data_augmentations.compute(var)
    print(output)
    assert output["input"].shape == (1, 3, 32, 32), "The 'input' tensor shape should be (1, 3, 32, 32)"


def test_noise_data_augmentation(data_augmentation_config):
    """Test NoiseDataAugmentation."""
    data_augmentations = DataAugmentations(augmentations_configs=data_augmentation_config)

    data1 = np.random.rand(20, 20, 1)
    var = {"input": data1}
    output = data_augmentations.compute(var)

    assert not np.array_equal(data1, output["input"]), "Noise should have modified the data"

    tensor = torch.randn(1, 3, 32, 32)
    var = {"input": tensor}
    output = data_augmentations.compute(var)

    assert output["input"].shape == tensor.shape, "The 'input' tensor shape should be (1, 3, 32, 32)"
    assert not torch.equal(tensor, output["input"]), "Noise should have modified the tensor"


def test_wrong_configuration(data_augmentation_config):
    """Test handling of incorrect configurations."""

    # Incorrect configuration for noise augmentation
    config = [
        {
            "type": "NoiseDataAugmentation",
            "name": "input",
            # 'keys_to_augment' is missing
        },
    ]

    with pytest.raises(TypeError):
        data_augmentations = DataAugmentations(augmentations_configs=config)
        data1 = np.random.rand(20, 20, 1)
        var = {"input": data1}
        data_augmentations.compute(var)


def test_multiple_augmentations(data_augmentation_config):
    """Test multiple augmentations (with several transformations)."""
    config = [
        {"type": "ToTensor", "force_device": "cpu"},
        {"type": "RandomHorizontalFlip", "p": 1.0},
        {"type": "NoiseDataAugmentation", "name": "input", "keys_to_augment": ["input"]},
    ]
    data_augmentations = DataAugmentations(augmentations_configs=config)

    data1 = np.random.rand(1, 20, 20)
    var = {"input": data1}
    output = data_augmentations.compute(var)

    assert isinstance(output["input"], torch.Tensor), "The 'input' result should be a tensor"
    assert output["input"].shape == (1, 20, 20), "The 'input' tensor shape should be (1, 20, 20)"
    assert not np.array_equal(data1, output["input"]), "The data should not be the same after noise augmentation"


def test_configure_combine_augmentation():
    """Test configuration with multiple combined transforms."""
    config = [
        {"type": "ToTensor", "force_device": "cpu"},
        {"type": "RandomHorizontalFlip", "p": 1.0},
        {"type": "Normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    ]
    data_augmentations = DataAugmentations(augmentations_configs=config)

    data_np = np.random.rand(3, 32, 32)
    var = {"input": data_np}
    output = data_augmentations.compute(var)

    assert isinstance(output["input"], torch.Tensor), "The result should be a tensor"
    assert output["input"].shape == (3, 32, 32), "The tensor shape should be (C, H, W)"
    assert output["input"].mean().item() == pytest.approx(0, abs=0.1), "The tensor mean should be close to 0"


def test_invalid_augmentation_type():
    """Test an invalid augmentation type."""
    config = [
        {"type": "InvalidType"},  # Unknown transformation type
    ]
    with pytest.raises(ValueError):
        data_augmentations = DataAugmentations(augmentations_configs=config)
        data_augmentations.compute({"input": np.random.rand(20, 20, 1)})
