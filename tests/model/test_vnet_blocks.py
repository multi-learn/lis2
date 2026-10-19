import pytest
import torch

from lis2.models.vnet import (
    ContBatchNorm2d,
    DownTransition,
    LUConv,
    UpTransition,
    elu_cons,
    passthrough,
)


def test_activation_factory_selects_elu_or_prelu():
    assert isinstance(elu_cons(True, 4), torch.nn.ELU)
    assert isinstance(elu_cons(False, 4), torch.nn.PReLU)


def test_passthrough_returns_input():
    tensor = torch.randn(1, 2)

    assert passthrough(tensor) is tensor


def test_cont_batch_norm_checks_input_dimensions():
    batch_norm = ContBatchNorm2d(2)
    with pytest.raises(ValueError, match="expected 4D input"):
        batch_norm(torch.randn(2, 3, 4))


def test_luconv_preserves_shape_with_elu_and_prelu():
    for use_elu in (True, False):
        layer = LUConv(2, use_elu)
        output = layer(torch.randn(2, 2, 8, 8))

        assert output.shape == (2, 2, 8, 8)


@pytest.mark.parametrize("dropout", [True, False])
def test_down_transition_halves_spatial_dimensions(dropout):
    layer = DownTransition(16, n_convs=1, elu=True, dropout=dropout)

    assert layer(torch.randn(2, 16, 16, 16)).shape == (2, 32, 8, 8)


@pytest.mark.parametrize("dropout", [True, False])
def test_up_transition_doubles_spatial_dimensions(dropout):
    layer = UpTransition(32, 32, n_convs=1, elu=True, dropout=dropout)

    output = layer(torch.randn(2, 32, 4, 4), torch.randn(2, 16, 8, 8))

    assert output.shape == (2, 32, 8, 8)
