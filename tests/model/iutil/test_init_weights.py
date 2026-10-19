import pytest
import torch
from torch import nn

from lis2.models.utils.init_weights import init_weights


@pytest.mark.parametrize("init_type", ["normal", "xavier", "kaiming", "orthogonal"])
def test_init_weights_initializes_supported_layers(init_type):
    network = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=3),
        nn.Linear(5, 3),
        nn.BatchNorm2d(4),
    )
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(5)

    init_weights(network, init_type)

    assert not torch.all(network[0].weight == 5)
    assert not torch.all(network[1].weight == 5)
    assert not torch.all(network[2].weight == 5)
    assert torch.all(network[2].bias == 0)


def test_init_weights_rejects_unknown_initialization():
    with pytest.raises(NotImplementedError, match="not implemented"):
        init_weights(nn.Linear(2, 2), "unknown")
