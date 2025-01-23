from enum import Enum
from typing import Literal

import torch
from torch import nn

from PNRIA.configs.config import Schema, Config
from PNRIA.torch_c.models.custom_model import BaseModel


class LayerFactory:
    CONV_LAYER_DICT = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    POOL_LAYER_DICT = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
    UPCONV_LAYER_DICT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
    BATCHNORM_LAYER_DICT = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    @staticmethod
    def get_conv_layer(dim, in_channels, out_channels, kernel_size=3, padding=1, **kwargs):
        if dim not in LayerFactory.CONV_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.CONV_LAYER_DICT[dim](
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs
        )

    @staticmethod
    def get_pool_layer(dim, kernel_size=2, stride=2, **kwargs):
        if dim not in LayerFactory.POOL_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.POOL_LAYER_DICT[dim](kernel_size=kernel_size, stride=stride, **kwargs)

    @staticmethod
    def get_upconv_layer(dim, in_channels, out_channels, kernel_size=2, stride=2, **kwargs):
        if dim not in LayerFactory.UPCONV_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.UPCONV_LAYER_DICT[dim](
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, **kwargs
        )

    @staticmethod
    def get_batchnorm_layer(dim, num_features):
        if dim not in LayerFactory.BATCHNORM_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.BATCHNORM_LAYER_DICT[dim](num_features=num_features)


class encoder_pos(Enum):
    BEFORE = "before"
    MIDDLE = "middle"
    AFTER = "after"


class BaseUNet(BaseModel):
    """
    A base class for UNet implementations with customizable blocks and position encoder.
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

    def __init__(
            self, in_channels, out_channels, features, num_blocks, dim, encoder=None, encoder_cat_position=None
    ):
        super(BaseUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_blocks = num_blocks
        self.dim = dim
        self.encoder = encoder
        self.encoder_cat_position = encoder_cat_position
        self.use_pe = encoder is not None

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.init_layers()

    def init_layers(self):

        t_features = self.features
        for i in range(self.num_blocks):
            in_ch = self.in_channels if i == 0 else t_features
            out_ch = t_features if i == 0 else t_features * 2
            self.encoders.append(self._create_block(in_ch, out_ch))
            self.pools.append(LayerFactory.get_pool_layer(self.dim))
            t_features = out_ch
        self.bottleneck = self._create_block(t_features, t_features * 2)
        t_features *= 2
        for i in range(self.num_blocks):
            t_features //= 2
            self.upconvs.append(LayerFactory.get_upconv_layer(self.dim, t_features * 2, t_features))
            self.decoders.append(self._create_block(t_features * 2, t_features))

        self.conv = LayerFactory.get_conv_layer(self.dim, self.features, self.out_channels, kernel_size=1, padding=0)
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
            )
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

    def _create_block(self, in_channels, out_channels):
        """
        Create a UNet block with two convolutional layers, batch normalization, and ReLU activation.
        """
        return nn.Sequential(
            LayerFactory.get_conv_layer(self.dim, in_channels, out_channels),
            LayerFactory.get_batchnorm_layer(self.dim, out_channels),
            nn.ReLU(inplace=True),
            LayerFactory.get_conv_layer(self.dim, out_channels, out_channels),
            LayerFactory.get_batchnorm_layer(self.dim, out_channels),
            nn.ReLU(inplace=True),
        )
