from enum import Enum
from typing import Optional, Tuple, Union, Literal

import torch
from configurable import Schema, Config
from torch import nn

from .base_model import BaseModel
from .encoder import BaseEncoder


class LayerFactory:
    """
    Factory class for creating various types of layers with specified dimensions.
    """

    CONV_LAYER_DICT = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    POOL_LAYER_DICT = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
    UPCONV_LAYER_DICT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}
    BATCHNORM_LAYER_DICT = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    @staticmethod
    def get_conv_layer(dim: int, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                       **kwargs) -> nn.Module:
        """
        Get a convolutional layer for the specified dimension.

        Args:
            dim (int): Dimension of the convolution (1, 2, or 3).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding for the convolution.
            **kwargs: Additional keyword arguments for the convolutional layer.

        Returns:
            nn.Module: The convolutional layer.
        """
        if dim not in LayerFactory.CONV_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.CONV_LAYER_DICT[dim](
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs
        )

    @staticmethod
    def get_pool_layer(dim: int, kernel_size: int = 2, stride: int = 2, **kwargs) -> nn.Module:
        """
        Get a pooling layer for the specified dimension.

        Args:
            dim (int): Dimension of the pooling (1, 2, or 3).
            kernel_size (int): Size of the pooling kernel.
            stride (int): Stride for the pooling.
            **kwargs: Additional keyword arguments for the pooling layer.

        Returns:
            nn.Module: The pooling layer.
        """
        if dim not in LayerFactory.POOL_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.POOL_LAYER_DICT[dim](kernel_size=kernel_size, stride=stride, **kwargs)

    @staticmethod
    def get_upconv_layer(dim: int, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2,
                         **kwargs) -> nn.Module:
        """
        Get a transposed convolutional layer for the specified dimension.

        Args:
            dim (int): Dimension of the transposed convolution (1, 2, or 3).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the transposed convolutional kernel.
            stride (int): Stride for the transposed convolution.
            **kwargs: Additional keyword arguments for the transposed convolutional layer.

        Returns:
            nn.Module: The transposed convolutional layer.
        """
        if dim not in LayerFactory.UPCONV_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.UPCONV_LAYER_DICT[dim](
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, **kwargs
        )

    @staticmethod
    def get_batchnorm_layer(dim: int, num_features: int) -> nn.Module:
        """
        Get a batch normalization layer for the specified dimension.

        Args:
            dim (int): Dimension of the batch normalization (1, 2, or 3).
            num_features (int): Number of features for the batch normalization.

        Returns:
            nn.Module: The batch normalization layer.
        """
        if dim not in LayerFactory.BATCHNORM_LAYER_DICT:
            raise ValueError(f"Unsupported dimension: {dim}. Use 1, 2, or 3.")
        return LayerFactory.BATCHNORM_LAYER_DICT[dim](num_features=num_features)


class encoder_pos(Enum):
    """
    The ``encoder_pos`` enumeration specifies the position of the encoder in the U-Net architecture. The possible values are:

    - ``BEFORE``: The encoder is placed before the encoding process begins. This configuration concatenates the position encoding with the input tensor at the start of the forward pass.
    - ``MIDDLE``: The encoder is placed after the bottleneck layer. This configuration concatenates the position encoding with the tensor immediately after the bottleneck, influencing the upsampling path of the U-Net.
    - ``AFTER``: The encoder is placed after the decoding process is complete. This configuration concatenates the position encoding with the output tensor at the end of the forward pass.

    These positions determine when the position encoding is integrated into the U-Net model's forward pass, affecting how spatial information is utilized throughout the network.
    """
    BEFORE = "before"
    MIDDLE = "middle"
    AFTER = "after"


class BaseUNet(BaseModel):
    """
    BaseUNet for configurable UNet implementations with customizable blocks and position encoder.

    A base class for UNet implementations that allows configuration of various parameters such as input/output channels,
    features, number of blocks, dimensionality, and encoder settings. This class provides a flexible structure for
    building UNet models with different configurations.

    Configuration:
        - **name** (str): The name of the UNet model.
        - **in_channels** (int): Number of input channels.
        - **out_channels** (int): Number of output channels.
        - **features** (int): Number of features in the UNet.
        - **num_blocks** (int): Number of blocks in the UNet.
        - **dim** (int): Dimensionality of the UNet (e.g., 2D, 3D).
        - **encoder** (Config, optional): Configuration for the encoder (:ref:`BaseEncoder`). Default is None.
        - **encoder_cat_position** (Literal["before", "middle", "after"], optional): Position to concatenate the encoder output. Default is "before".

        Example Configuration:
            .. code-block:: python

                config = {
                    "name": "example_unet",
                    "in_channels": 3,
                    "out_channels": 1,
                    "features": 64,
                    "num_blocks": 4,
                    "dim": 2,
                    "encoder": "configs/encoder/encoderLin.yml",
                    "encoder_cat_position": "before"
                }

        Aliases:
            - `unet`
            - `base_unet`
        """

    aliases = ["unet", "base_unet"]

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


    def __init__(self):
        super(BaseUNet, self).__init__()
        self.use_pe = False
        if self.encoder is not None:
            self.encoder = BaseEncoder.from_config(self.encoder)
            self.use_pe = True
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.init_layers()

    def init_layers(self):
        """
        Initialize the layers of the UNet model.
        """
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

    def core_forward(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> torch.Tensor:
        """
        Core forward pass of the UNet model.

        Args:
            batch (Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]): Input batch, which can be a tensor or a tuple containing the tensor and position encoding.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        x, pe = batch if isinstance(batch, tuple) else (batch, None)
        assert isinstance(x, torch.Tensor), "Input must be a tensor."
        assert self.use_pe != (pe is None), "Position encoding is not configured properly."
        enc_outputs = []
        if self.use_pe and self.encoder_cat_position == encoder_pos.BEFORE:
            x = torch.cat((x, pe), dim=1)

        for i in range(self.num_blocks):
            x = self.encoders[i](x)
            enc_outputs.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)
        if self.use_pe and self.encoder_cat_position == encoder_pos.MIDDLE:
            x = torch.cat((x, pe), dim=1)

        for i in range(self.num_blocks):
            x = self.upconvs[i](x)
            x = torch.cat((x, enc_outputs[self.num_blocks - i - 1]), dim=1)
            x = self.decoders[i](x)

        if self.use_pe and self.encoder_cat_position == encoder_pos.AFTER:
            x = torch.cat((x, pe), dim=1)
        return torch.sigmoid(self.conv(x))

    def preprocess_forward(self, patch: torch.Tensor, positions: Optional[torch.Tensor] = None, **kwargs) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess the input data before the core forward pass.

        Args:
            patch (torch.Tensor): Input tensor.
            positions (Optional[torch.Tensor]): Position encoding tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The preprocessed input tensor and position encoding.
        """
        # log
        self.logger.info(f"patch shape: {patch.shape}")
        self.logger.info(f"positions shape: {positions.shape}")
        self.logger.info(f"positions: {positions}")
        self.logger.info(f"use_pe: {self.use_pe}")

        assert self.use_pe != (positions is None), "Model has position encoding but no position is provided."
        pe = self.encoder(positions) if self.use_pe else None
        return patch, pe

    def _create_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a UNet block with two convolutional layers, batch normalization, and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: The created UNet block.
        """
        return nn.Sequential(
            LayerFactory.get_conv_layer(self.dim, in_channels, out_channels),
            LayerFactory.get_batchnorm_layer(self.dim, out_channels),
            nn.ReLU(inplace=True),
            LayerFactory.get_conv_layer(self.dim, out_channels, out_channels),
            LayerFactory.get_batchnorm_layer(self.dim, out_channels),
            nn.ReLU(inplace=True),
        )

    def postprocess_forward(self, output: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Postprocess the output data after the core forward pass.

        Args:
            output (torch.Tensor): Output tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The postprocessed output tensor.
        """
        return output
