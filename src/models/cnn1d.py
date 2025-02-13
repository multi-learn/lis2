import torch
from configurable import Schema
from torch import nn

from .base_model import BaseModel


class CNN1D(BaseModel):
    """
    1D CNN model for processing one-dimensional data.

    This class implements a 1D Convolutional Neural Network (CNN) model designed to process one-dimensional data.
    It includes a sequence of convolutional layers, ReLU activations, pooling, dropout, and fully connected layers.

    Configuration:

    name (str): The name of the 1D CNN model.
    in_channels (int): Number of input channels. Default is 1.
    out_channels (int): Number of output channels. Default is 1.
    kernel_size (int): Size of the convolutional kernel. Default is 16.
    stride (int): Stride for the convolution. Default is 2.
    padding (int): Padding for the convolution. Default is 0.
    dropout_rate (float): Dropout rate for the dropout layers. Default is 0.5.
    linear_features (int): Number of features in the linear layers. Default is 64.

    Example Configuration (Python):
        .. code-block:: python

            config = {
                "name": "example_cnn1d",
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 16,
                "stride": 2,
                "padding": 0,
                "dropout_rate": 0.5,
                "linear_features": 64
            }

    Aliases:

    cnn1d
    """

    config_schema = {
        "in_channels": Schema(int, default=1),
        "out_channels": Schema(int, default=1),
        "kernel_size": Schema(int, default=16),
        "stride": Schema(int, default=2),
        "padding": Schema(int, default=0),
        "dropout_rate": Schema(float, default=0.5),
        "linear_features": Schema(int, default=64),
    }

    def __init__(self, *args, **kwargs):
        super(CNN1D, self).__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels, out_channels=64, kernel_size=self.kernel_size, stride=self.stride,
                padding=self.padding
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=387, stride=1),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=64, out_features=self.linear_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=self.linear_features, out_features=self.out_channels),
        )

    def _core_forward(self, x):
        """
        Core forward pass of the 1D CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        x = torch.unsqueeze(x, dim=1)
        output = self.layers(x)
        return torch.squeeze(output, dim=1)

    def _preprocess_forward(self, patch, *args, **kwargs):
        """
        Preprocess the input data before the core forward pass.

        Args:
            patch (torch.Tensor): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The preprocessed input tensor.
        """
        return patch
