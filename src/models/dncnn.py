"""DnCNN implementation.

Code from https://github.com/yjn870/DnCNN-pytorch (https://arxiv.org/abs/1608.03981)
"""

from configurable import Schema
from torch import nn

from src.models.base_model import BaseModel


class DnCNN(BaseModel):
    """
    DnCNN model

    Attributes
    ----------
    num_layers: int, optional
        Number of layers in the model (default: 10)
    num_features: int, optional
        Number of features in the convolutional layers (default: 64)
    """

    config_schema = {
        "num_layers": Schema(int, default=10),
        "num_features": Schema(int, default=64),
    }

    def __init__(self, *args, **kwargs):
        super(DnCNN, self).__init__(*args, **kwargs)

        layers = [
            nn.Sequential(
                nn.Conv2d(1, self.num_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
        ]
        for _ in range(self.num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.num_features, self.num_features, kernel_size=3, padding=1
                    ),
                    nn.BatchNorm2d(self.num_features),
                    nn.ReLU(inplace=True),
                )
            )
        layers.append(nn.Conv2d(self.num_features, 1, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _core_forward(self, batch):
        return batch - self.layers(batch)

    def _preprocess_forward(self, patch, *args, **kwargs):
        return patch
