import pytest
import torch

from lis2.models import BaseModel


@pytest.mark.parametrize("use_batch_norm", [True, False])
def test_dncnn_forward_preserves_image_shape(use_batch_norm):
    model = BaseModel.from_config(
        {
            "type": "DnCNN",
            "num_layers": 3,
            "num_features": 4,
            "use_batch_norm": use_batch_norm,
        }
    )
    model.eval()

    output = model(torch.randn(2, 1, 8, 8))

    assert output.shape == (2, 1, 8, 8)


def test_dncnn_rejects_unsupported_activation():
    with pytest.raises(RuntimeError, match="Error while instantiating"):
        BaseModel.from_config(
            {
                "type": "DnCNN",
                "num_layers": 3,
                "num_features": 4,
                "activation": "unsupported",
            }
        )
