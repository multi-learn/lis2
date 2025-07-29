"""
UNetPlusPlus network
Code from https://github.com/ZJUGiveLab/UNet-Version (no licence)

Clarification and improvement by François-Xavier Dupé
"""
from typing import Any

import torch
import torch.nn as nn
from configurable import Schema

from .base_model import BaseModel
from .utils.init_weights import init_weights
from .utils.layers import unetConv2, unetUp_origin


class UNetPP(BaseModel):
    """
    UNetPlusPlus model for image segmentation.

    This class implements the UNet++ architecture, which is an enhanced version of the UNet model designed for image segmentation tasks.
    It includes additional skip connections and deep supervision to improve performance.

    Configuration:
    name (str): The name of the UNet++ model.
    in_channels (int): Number of input channels. Default is 1.
    n_classes (int): Number of output channels. Default is 1.
    feature_scale (int): Scale factor for the number of features in each layer. Default is 4.
    is_deconv (bool): Whether to use deconvolution layers for upsampling. Default is True.
    is_batchnorm (bool): Whether to use batch normalization layers. Default is True.
    is_ds (bool): Whether to use deep supervision. Default is True.
    filters (list): List of filter sizes for each layer. Default is [64, 128, 256, 512, 1024].
    kernel_size (int): Size of the convolutional kernel. Default is 3.
    padding (int): Padding for the convolution. Default is 1.
    activation (str): Activation function to use. Default is 'relu'.

    Example Configuration (Python):
        .. code-block:: python

            config = {
                "name": "example_unetpp",
                "in_channels": 1,
                "n_classes": 1,
                "feature_scale": 4,
                "is_deconv": True,
                "is_batchnorm": True,
                "is_ds": True,
                "filters": [64, 128, 256, 512, 1024],
                "kernel_size": 3,
                "padding": 1,
                "activation": "relu"
            }

    Aliases:

    unetpp
    """

    aliases = ["unetpp"]

    config_schema = {
        "in_channels": Schema(int, default=1),
        "n_classes": Schema(int, default=1),
        "feature_scale": Schema(int, default=4),
        "is_deconv": Schema(bool, default=True),
        "is_batchnorm": Schema(bool, default=True),
        "is_ds": Schema(bool, default=True),
        "filters": Schema(list, default=[64, 128, 256, 512, 1024]),
        "kernel_size": Schema(int, default=3),
        "padding": Schema(int, default=1),
        "activation": Schema(str, default="relu"),
    }

    def __init__(self):
        super(UNetPP, self).__init__()

        self._init_layers()

    def _init_layers(self):
        # Downsampling
        self.conv00 = unetConv2(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(self.filters[3], self.filters[4], self.is_batchnorm)

        # Upsampling
        self.up_concat01 = unetUp_origin(self.filters[1], self.filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(self.filters[2], self.filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(self.filters[3], self.filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(self.filters[4], self.filters[3], self.is_deconv)
        self.up_concat02 = unetUp_origin(self.filters[1], self.filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(self.filters[2], self.filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(self.filters[3], self.filters[2], self.is_deconv, 3)
        self.up_concat03 = unetUp_origin(self.filters[1], self.filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(self.filters[2], self.filters[1], self.is_deconv, 4)
        self.up_concat04 = unetUp_origin(self.filters[1], self.filters[0], self.is_deconv, 5)

        # Final convolutions
        self.final_1 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        self.final_2 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        self.final_3 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        self.final_4 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def core_forward(self, x):
        # Column 0
        x_00 = self.conv00(x)
        maxpool0 = self.maxpool0(x_00)
        x_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(x_10)
        x_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(x_20)
        x_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(x_30)
        x_40 = self.conv40(maxpool3)

        # Column 1
        x_01 = self.up_concat01(x_10, x_00)
        x_11 = self.up_concat11(x_20, x_10)
        x_21 = self.up_concat21(x_30, x_20)
        x_31 = self.up_concat31(x_40, x_30)

        # Column 2
        x_02 = self.up_concat02(x_11, x_00, x_01)
        x_12 = self.up_concat12(x_21, x_10, x_11)
        x_22 = self.up_concat22(x_31, x_20, x_21)

        # Column 3
        x_03 = self.up_concat03(x_12, x_00, x_01, x_02)
        x_13 = self.up_concat13(x_22, x_10, x_11, x_12)

        # Column 4
        x_04 = self.up_concat04(x_13, x_00, x_01, x_02, x_03)

        # Final layer
        final_1 = self.final_1(x_01)
        final_2 = self.final_2(x_02)
        final_3 = self.final_3(x_03)
        final_4 = self.final_4(x_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return torch.sigmoid(final)
        else:
            return torch.sigmoid(final_4)

    def preprocess_forward(self, patch, *args, **kwargs):
        return patch

    def postprocess_forward(self, x: Any) -> torch.Tensor:
        return x
