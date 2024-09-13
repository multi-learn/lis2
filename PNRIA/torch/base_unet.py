from collections import OrderedDict
import torch
from torch import nn
from PNRIA.torch.custom_model import BaseModel

CONV_LAYER_DICT = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
POOL_LAYER_DICT = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
UPCONV_LAYER_DICT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

def get_conv_layer(dim, in_channels, out_channels, kernel_size, padding, bias):
    try:
        conv_layer = CONV_LAYER_DICT[dim]
    except KeyError:
        raise ValueError(f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3.")
    return conv_layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
    )

def get_pool_layer(dim, kernel_size, stride):
    try:
        pool_layer = POOL_LAYER_DICT[dim]
    except KeyError:
        raise ValueError(f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3.")
    return pool_layer(
        kernel_size=kernel_size,
        stride=stride,
    )

def get_upconv_layer(dim, in_channels, out_channels, kernel_size, stride):
    try:
        upconv_layer = UPCONV_LAYER_DICT[dim]
    except KeyError:
        raise ValueError(f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3.")
    return upconv_layer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )

class BaseUNet(BaseModel):
    """
    A base class for UNet implementations with configurable number of blocks.
    """

    required_keys = ["in_channels", "out_channels", "features", "num_blocks", "dim"]

    _CONV_KERNEL_SIZE = 3
    _CONV_PADDING = 1
    _MAX_POOL_KERNEL_SIZE = 2
    _MAX_POOL_STRIDE = 2
    _UPCONV_KERNEL_SIZE = 2
    _UPCONV_STRIDE = 2

    def __init__(self):
        super(BaseUNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        self.init_layer()

    def init_layer(self):
        t_features = self.features
        # Create encoder layers
        for i in range(self.num_blocks):
            if i == 0:
                self.encoders.append(self._block(self.in_channels, t_features, f"enc{i + 1}"))
            else:
                self.encoders.append(self._block(t_features, t_features * 2, f"enc{i + 1}"))
                t_features *= 2
            self.pools.append(get_pool_layer(self.dim, self._MAX_POOL_KERNEL_SIZE, self._MAX_POOL_STRIDE))

        # Bottleneck
        self.bottleneck = self._block(t_features, t_features * 2, "bottleneck")

        # Create decoder layers
        for i in range(self.num_blocks, 0, -1):
            self.upconvs.append(get_upconv_layer(self.dim, t_features * 2, t_features, self._UPCONV_KERNEL_SIZE, self._UPCONV_STRIDE))
            self.decoders.append(self._block(t_features * 2, t_features, f"dec{i}"))
            t_features //= 2

        # Final convolution layer
        self.conv = get_conv_layer(self.dim, t_features, self.out_channels, self._CONV_KERNEL_SIZE, self._CONV_PADDING, False)


    def _core_forward(self, x):
        enc_outputs = []

        # Encoder forward pass
        for i in range(self.num_blocks):
            x = self.encoders[i](x)
            enc_outputs.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward pass with skip connections
        for i in range(self.num_blocks):
            x = self.upconvs[i](x)
            x = torch.cat((x, enc_outputs[self.num_blocks - i - 1]), dim=1)  # Skip connection
            x = self.decoders[i](x)

        # Final output
        return torch.sigmoid(self.conv(x))

    def _preprocess_forward(self, *args, **kwargs):
        # No preprocessing required in this case
        return args[0]

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        self._get_conv_layer(self.dim, in_channels, features),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        self._get_conv_layer(self.dim, features, features),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    # regino layer factory methods

    @staticmethod
    def _get_conv_layer(self, dim, in_channels, out_channels):
        try:
            conv_layer = CONV_LAYER_DICT[dim]
        except KeyError:
            raise ValueError(f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3.")
        return conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._CONV_KERNEL_SIZE,
            padding=self._CONV_PADDING,
            bias=False,
        )

    @staticmethod
    def _get_pool_layer(self, dim):
        try:
            pool_layer = POOL_LAYER_DICT[dim]
        except KeyError:
            raise ValueError(f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3.")
        return pool_layer(
            kernel_size=self._MAX_POOL_KERNEL_SIZE,
            stride=self._MAX_POOL_STRIDE,
        )

    @staticmethod
    def _get_upconv_layer(self, dim, in_channels, out_channels):
        try:
            upconv_layer = UPCONV_LAYER_DICT[dim]
        except KeyError:
            raise ValueError(f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3.")
        return upconv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._UPCONV_KERNEL_SIZE,
            stride=self._UPCONV_STRIDE,
        )
