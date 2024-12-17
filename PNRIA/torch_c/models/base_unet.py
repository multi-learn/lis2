from collections import OrderedDict
from enum import Enum
from typing import Literal

import torch
from torch import nn

from PNRIA.configs.config import Schema, Config
from PNRIA.torch_c.encoder import Encoder
from PNRIA.torch_c.models.custom_model import BaseModel

CONV_LAYER_DICT = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
POOL_LAYER_DICT = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
UPCONV_LAYER_DICT = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}


class encoder_pos(Enum):
    before = "before"
    middle = "middle"
    after = "after"


class BaseUNet(BaseModel):
    """
    A base class for UNet implementations with Customizable number of blocks and position encoder.
    """

    aliases = ["unet"]

    config_schema = {
        "in_channels": Schema(int),
        "out_channels": Schema(int),
        "features": Schema(int),
        "num_blocks": Schema(int),
        "dim": Schema(int, aliases=["dimension"]),
        "encoder": Schema(Config, optional=True),
        "encoder_cat_position": Schema(
            Literal["before", "middle", "after"], aliases=["encoder_pos"], optional=True
        ),
    }

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
        self.use_pe = False
        if self.encoder is not None:
            self.encoder = Encoder.from_config(self.encoder)
            self.use_pe = True

        self.init_layer()

    def preconditions(self):
        if hasattr(self, "encoder"):
            assert hasattr(
                self, "encoder_cat_position"
            ), "encoder_cat_position is required when encoder is provided."
        if hasattr(self, "encoder_cat_position"):
            assert hasattr(
                self, "encoder"
            ), "encoder is required when encoder_cat_position is provided."

    def init_layer(self):
        t_features = self.features
        # Create encoder layers
        for i in range(self.num_blocks):
            if i == 0:
                self.encoders.append(
                    self._block(self.in_channels, t_features, f"enc{i + 1}")
                )
            else:
                self.encoders.append(
                    self._block(t_features, t_features * 2, f"enc{i + 1}")
                )
                t_features *= 2
            self.pools.append(self._get_pool_layer(self.dim))
        self.bottleneck = self._block(t_features, t_features * 2, "bottleneck")
        t_features *= 2
        for i in range(self.num_blocks, 0, -1):
            t_features //= 2
            self.upconvs.append(
                self._get_upconv_layer(self.dim, t_features * 2, t_features)
            )
            self.decoders.append(self._block(t_features * 2, t_features, f"dec{i}"))

        self.conv = nn.Conv2d(
            in_channels=self.features, out_channels=self.out_channels, kernel_size=1
        )

    def _core_forward(self, batch):

        x, pe = batch if isinstance(batch, tuple) else (batch, None)
        assert isinstance(x, torch.Tensor), "Input must be a tensor."
        assert self.use_pe != (
            pe is None
        ), "Position encoding is not configured properly."
        enc_outputs = []
        if self.use_pe and self.encoder_cat_position == encoder_pos.before:
            x = torch.cat((x, pe), dim=1)

        for i in range(self.num_blocks):
            x = self.encoders[i](x)
            enc_outputs.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)
        if self.use_pe and self.encoder_cat_position == encoder_pos.middle:
            x = torch.cat((x, pe), dim=1)

        for i in range(self.num_blocks):
            x = self.upconvs[i](x)
            x = torch.cat(
                (x, enc_outputs[self.num_blocks - i - 1]), dim=1
            )  # Skip connection
            x = self.decoders[i](x)

        if self.use_pe and self.encoder_cat_position == encoder_pos.after:
            x = torch.cat((x, pe), dim=1)
        return torch.sigmoid(self.conv(x))

    def _preprocess_forward(self, patch, positions=None, **kwargs):
        assert self.use_pe != (
            positions is None
        ), "Model has position encoding but no position is provided."
        pe = self.encoder(positions) if self.use_pe else None
        return patch, pe

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

    # region layer factory methods

    def _get_conv_layer(self, dim, in_channels, out_channels):
        try:
            conv_layer = CONV_LAYER_DICT[dim]
        except KeyError:
            raise ValueError(
                f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3."
            )
        return conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._CONV_KERNEL_SIZE,
            padding=self._CONV_PADDING,
            bias=False,
        )

    def _get_pool_layer(self, dim):
        try:
            pool_layer = POOL_LAYER_DICT[dim]
        except KeyError:
            raise ValueError(
                f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3."
            )
        return pool_layer(
            kernel_size=self._MAX_POOL_KERNEL_SIZE,
            stride=self._MAX_POOL_STRIDE,
        )

    def _get_upconv_layer(self, dim, in_channels, out_channels):
        try:
            upconv_layer = UPCONV_LAYER_DICT[dim]
        except KeyError:
            raise ValueError(
                f"Unsupported dimension: {dim}. Supported dimensions are 1, 2, and 3."
            )
        return upconv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self._UPCONV_KERNEL_SIZE,
            stride=self._UPCONV_STRIDE,
        )

    # endregion
