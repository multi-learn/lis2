"""1D CNN implementation.

blabla
"""

from configurable import Schema
from torch import nn

from .base_model import BaseModel


class VGG1D(BaseModel):
    """
    VGG1D model for processing one-dimensional data.

    This class implements a 1D version of the VGG architecture, which is designed for processing one-dimensional data.
    It includes multiple convolutional layers with batch normalization and ReLU activations, followed by fully connected layers.

    Configuration:
    name (str): The name of the VGG1D model.
    in_channels (int): Number of input channels. Default is 1.
    out_channels (int): Number of output channels. Default is 1.
    n_convs (list of int): Number of convolutions in each layer. Default is [2, 2, 3, 3, 3].
    fc_units (list of int): Number of units in the fully connected layers. Default is [4096, 4096].
    kernel_size (int): Size of the convolutional kernel. Default is 3.
    stride (int): Stride for the convolution. Default is 1.
    padding (int): Padding for the convolution. Default is 1.
    dropout_rate (float): Dropout rate for the dropout layers. Default is 0.5.
    activation (str): Activation function to use. Default is 'relu'.

    Example Configuration (Python):
        .. code-block:: python

            config = {
                "name": "example_vgg1d",
                "in_channels": 1,
                "out_channels": 1,
                "n_convs": [2, 2, 3, 3, 3],
                "fc_units": [4096, 4096],
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "dropout_rate": 0.5,
                "activation": "relu"
            }

    Aliases:

    vgg1d
    """

    config_schema = {
        "in_channels": Schema(int, default=1),
        "out_channels": Schema(int, default=1),
        "n_convs": Schema(list, default=[2, 2, 3, 3, 3]),
        "fc_units": Schema(list, default=[4096, 4096]),
        "kernel_size": Schema(int, default=3),
        "stride": Schema(int, default=1),
        "padding": Schema(int, default=1),
        "dropout_rate": Schema(float, default=0.5),
        "activation": Schema(str, default="relu"),
    }

    def __init__(self, *args, **kwargs):
        super(VGG1D, self).__init__(*args, **kwargs)

        layers = []
        in_channels = self.in_channels
        out_channels = 64
        for i in range(self.n_convs[0]):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                          padding=self.padding)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self._get_activation())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=1))

        out_channels = 128
        for i in range(self.n_convs[1]):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                          padding=self.padding)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self._get_activation())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        out_channels = 256
        for i in range(self.n_convs[2]):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                          padding=self.padding)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self._get_activation())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        out_channels = 512
        for i in range(self.n_convs[3]):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                          padding=self.padding)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self._get_activation())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        out_channels = 512
        for i in range(self.n_convs[4]):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                          padding=self.padding)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self._get_activation())
            in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        layers.append(nn.Flatten())
        layers.append(nn.Dropout(self.dropout_rate))

        in_features = 2 * 7 * 512
        for units in self.fc_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            in_features = units

        layers.append(nn.Linear(in_features, self.out_channels))

        self.layers = nn.Sequential(*layers)

    def _get_activation(self):
        """
        Get the activation function based on the configuration.

        Returns:
            nn.Module: The activation function.
        """
        if self.activation.lower() == "relu":
            return nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def core_forward(self, x):
        """
        Core forward pass of the VGG1D model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        output = self.layers(x)
        return output

    def preprocess_forward(self, patch, *args, **kwargs):
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
