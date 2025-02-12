"""1D CNN implementation.

blabla
"""

import torch
from torch import nn

from src.models.custom_model import BaseModel


class CNN1D(BaseModel):
    """
    1D CNN model

    Attributes
    ----------
    None
    """

    config_schema = {}

    def __init__(self, *args, **kwargs):
        super(CNN1D, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=1, out_channels=64, kernel_size=16, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv1d(
                    in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=387, stride=1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=1),
            ]
        )

    def _core_forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        output = self.layers(x)
        return torch.squeeze(output, dim=1)

    def _preprocess_forward(self, patch, *args, **kwargs):
        return patch
