import pytest
import torch

from lis2.models import BaseModel
from lis2.models.vgg1d import VGG1D


class _TestableVGG1D(VGG1D):
    def postprocess_forward(self, x):
        return x


def test_cnn1d_forward_runs_preprocessing_core_and_postprocessing():
    model = BaseModel.from_config(
        {
            "type": "CNN1D",
            "linear_features": 8,
            "dropout_rate": 0.0,
        }
    )
    model.eval()

    output = model(torch.randn(2, 1, 788))

    assert output.shape == (2,)


@pytest.mark.parametrize("activation", ["relu", "unsupported"])
def test_vgg1d_activation_configuration(activation):
    if activation == "unsupported":
        with pytest.raises(RuntimeError, match="Error while instantiating"):
            _TestableVGG1D._from_config(
                {
                    "type": "_TestableVGG1D",
                    "n_convs": [1, 1, 1, 1, 1],
                    "fc_units": [8],
                    "dropout_rate": 0.0,
                    "activation": activation,
                }
            )
        return

    model = _TestableVGG1D._from_config(
        {
            "type": "_TestableVGG1D",
            "n_convs": [1, 1, 1, 1, 1],
            "fc_units": [8],
            "dropout_rate": 0.0,
            "activation": activation,
        }
        )
    model.eval()

    output = model(torch.randn(2, 1, 240))

    assert output.shape == (2, 1)
