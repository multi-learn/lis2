import pytest
import torch

from lis2.models import BaseModel


@pytest.mark.parametrize(
    ("is_deconv", "is_ds"),
    [(True, True), (True, False)],
)
def test_unetpp_forward_supports_upsampling_and_supervision_modes(is_deconv, is_ds):
    model = BaseModel.from_config(
        {
            "type": "UNetPP",
            "in_channels": 1,
            "n_classes": 2,
            "feature_scale": 1,
            "filters": [2, 4, 8, 16, 32],
            "is_deconv": is_deconv,
            "is_batchnorm": False,
            "is_ds": is_ds,
        }
    )
    model.eval()

    output = model(torch.randn(1, 1, 32, 32))

    assert output.shape == (1, 2, 32, 32)
    assert torch.all((output >= 0) & (output <= 1))
