"""1D CNN implementation.

blabla
"""
from configurable import Schema
from torch import nn

from models.custom_model import BaseModel


class VGG1D(BaseModel):
    """
    VGG1D model

    Attributes
    ----------
    in_channels: int, optional
        Number of input channels (default: 1)
    out_channels: int, optional
        Number of output channels (default: 1)
    n_convs: list of int, optional
        Number of convolutions in each layer (default: [2, 2, 3, 3, 3])
    fc_units: list of int, optional
        Number of units in the fully connected layers (default: [4096, 4096])
    """

    config_schema = {
        "in_channels": Schema(int, default=1),
        "out_channels": Schema(int, default=1),
        "n_convs": Schema(list, default=[2, 2, 3, 3, 3]),
        "fc_units": Schema(list, default=[4096, 4096]),
    }

    def __init__(self, *args, **kwargs):
        super(VGG1D, self).__init__(*args, **kwargs)

        layers = []
        in_channels = self.in_channels
        out_channels = 64
        for i in range(self.n_convs[0]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=1))

        out_channels = 128
        for i in range(self.n_convs[1]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        out_channels = 256
        for i in range(self.n_convs[2]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        out_channels = 512
        for i in range(self.n_convs[3]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        out_channels = 512
        for i in range(self.n_convs[4]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        layers.append(nn.Flatten())
        layers.append(nn.Dropout(0.5))

        in_features = 2 * 7 * 512
        for units in self.fc_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_features = units

        layers.append(nn.Linear(in_features, self.out_channels))

        self.layers = nn.Sequential(*layers)

    def _core_forward(self, x):
        output = self.layers(x)
        return output

    def _preprocess_forward(self, patch, *args, **kwargs):
        return patch
