"""DnCNN implementation.

Code from https://github.com/yjn870/DnCNN-pytorch (https://arxiv.org/abs/1608.03981)
"""

from configurable import Schema
from torch import nn

from .base_model import BaseModel


class DnCNN(BaseModel):
    """
    DnCNN model for image denoising.

    This class implements the DnCNN (Denoising Convolutional Neural Network) model, which is designed for image denoising tasks.
    The model consists of multiple convolutional layers with ReLU activations and batch normalization.

    Configuration:
    name (str): The name of the DnCNN model.
    num_layers (int): Number of layers in the model. Default is 10.
    num_features (int): Number of features in the convolutional layers. Default is 64.
    kernel_size (int): Size of the convolutional kernel. Default is 3.
    stride (int): Stride for the convolution. Default is 1.
    padding (int): Padding for the convolution. Default is 1.
    use_batch_norm (bool): Whether to use batch normalization. Default is True.
    activation (str): Activation function to use. Default is 'relu'.

    Example Configuration (Python):
        .. code-block:: python

            config = {
                "name": "example_dncnn",
                "num_layers": 10,
                "num_features": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "use_batch_norm": True,
                "activation": "relu"
            }

    Aliases:

    dncnn
    denoising_cnn
    """

    config_schema = {
        "num_layers": Schema(int, default=10),
        "num_features": Schema(int, default=64),
        "kernel_size": Schema(int, default=3),
        "stride": Schema(int, default=1),
        "padding": Schema(int, default=1),
        "use_batch_norm": Schema(bool, default=True),
        "activation": Schema(str, default="relu"),
    }

    def __init__(self, *args, **kwargs):
        super(DnCNN, self).__init__(*args, **kwargs)

        layers = [
            nn.Sequential(
                nn.Conv2d(1, self.num_features, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
                self._get_activation(),
            )
        ]
        for _ in range(self.num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.num_features, self.num_features, kernel_size=self.kernel_size, padding=self.padding
                    ),
                    nn.BatchNorm2d(self.num_features) if self.use_batch_norm else nn.Identity(),
                    self._get_activation(),
                )
            )
        layers.append(nn.Conv2d(self.num_features, 1, kernel_size=self.kernel_size, padding=self.padding))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def core_forward(self, batch):
        """
        Core forward pass of the DnCNN model.

        Args:
            batch (torch.Tensor): Input batch tensor.

        Returns:
            torch.Tensor: The denoised output tensor.
        """
        return self.layers(batch)

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

    def postprocess_forward(self, patch, *args, **kwargs):
        """
        Postprocess the input data after the core forward pass.

        Args:
            patch (torch.Tensor): Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The postprocessed output tensor.
        """
        return patch
