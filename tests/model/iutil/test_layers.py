import pytest
import torch

from lis2.models.utils.layers import unetConv2, unetUp, unetUp_origin


@pytest.mark.parametrize("is_batchnorm", [True, False])
def test_unet_conv2_preserves_spatial_shape(is_batchnorm):
    layer = unetConv2(1, 2, is_batchnorm=is_batchnorm, n=2)

    assert layer(torch.randn(1, 1, 8, 8)).shape == (1, 2, 8, 8)


@pytest.mark.parametrize("is_deconv", [True, False])
def test_unet_up_upsamples_and_concatenates_skip(is_deconv):
    layer = unetUp(2, 2, is_deconv=is_deconv)

    output = layer(torch.randn(1, 2, 4, 4), torch.randn(1, 2, 8, 8))

    assert output.shape == (1, 2, 8, 8)


def test_unet_up_origin_with_transposed_convolution():
    layer = unetUp_origin(4, 2, is_deconv=True)

    output = layer(torch.randn(1, 4, 4, 4), torch.randn(1, 2, 8, 8))

    assert output.shape == (1, 2, 8, 8)


def test_unet_up_origin_with_interpolation_and_multiple_skips():
    layer = unetUp_origin(2, 2, is_deconv=False, n_concat=4)

    output = layer(
        torch.randn(1, 2, 4, 4),
        torch.randn(1, 2, 8, 8),
        torch.randn(1, 2, 8, 8),
    )

    assert output.shape == (1, 2, 8, 8)
