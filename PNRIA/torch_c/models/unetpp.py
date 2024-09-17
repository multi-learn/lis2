"""
UNetPlusPlus network
Code from https://github.com/ZJUGiveLab/UNet-Version (no licence)

Clarification and improvement by François-Xavier Dupé
"""
import torch
import torch.nn as nn

from PNRIA.torch_c.models.custom_model import BaseModel
from deep_filaments.torch.third_party.layers import unetConv2, unetUp_origin
from deep_filaments.torch.third_party.init_weights import init_weights


class UNetPP(BaseModel):
    """
    UNetPlusPlus model

    Attributes
    ----------
    in_channels: int, optional
        The number of input channels (default: 1)
    self.n_classes: int, optional
        The number of output channels (default: 1)
    feature_scale: int, optional
        The scale factor for the number of features in each layer (default: 4)
    is_deconv: bool, optional
        Whether to use deconvolution layers for upsampling (default: True)
    is_batchnorm: bool, optional
        Whether to use batch normalization layers (default: True)
    is_ds: bool, optional
        Whether to use deep supervision (default: True)
    """

    aliases = ["unetpp"]

    config_schema = {
        "in_channels": {"type": int, "default": 1},
        "n_classes": {"type": int, "default": 1},
        "feature_scale": {"type": int, "default": 4},
        "is_deconv": {"type": bool, "default": True},
        "is_batchnorm": {"type": bool, "default": True},
        "is_ds": {"type": bool, "default": True},
        "filters": {"type": list, "default": [64, 128, 256, 512, 1024]},
    }

    def __init__(
        self):
        super(UNetPP, self).__init__()

        self._init_layers()

    def _init_layers(self):
        # downsampling
        self.conv00 = unetConv2(self.in_channels, self.filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(self.filters[0], self.filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(self.filters[1], self.filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(self.filters[2], self.filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(self.filters[3], self.filters[4], self.is_batchnorm)
        # upsampling
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
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        self.final_2 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        self.final_3 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        self.final_4 = nn.Conv2d(self.filters[0], self.n_classes, (1, 1))
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def _core_forward(self, x):
        # column : 0
        x_00 = self.conv00(x)
        maxpool0 = self.maxpool0(x_00)
        x_10 = self.conv10(maxpool0)
        maxpool1 = self.maxpool1(x_10)
        x_20 = self.conv20(maxpool1)
        maxpool2 = self.maxpool2(x_20)
        x_30 = self.conv30(maxpool2)
        maxpool3 = self.maxpool3(x_30)
        x_40 = self.conv40(maxpool3)

        # column : 1
        x_01 = self.up_concat01(x_10, x_00)
        x_11 = self.up_concat11(x_20, x_10)
        x_21 = self.up_concat21(x_30, x_20)
        x_31 = self.up_concat31(x_40, x_30)
        # column : 2
        x_02 = self.up_concat02(x_11, x_00, x_01)
        x_12 = self.up_concat12(x_21, x_10, x_11)
        x_22 = self.up_concat22(x_31, x_20, x_21)
        # column : 3
        x_03 = self.up_concat03(x_12, x_00, x_01, x_02)
        x_13 = self.up_concat13(x_22, x_10, x_11, x_12)
        # column : 4
        x_04 = self.up_concat04(x_13, x_00, x_01, x_02, x_03)

        # final layer
        final_1 = self.final_1(x_01)
        final_2 = self.final_2(x_02)
        final_3 = self.final_3(x_03)
        final_4 = self.final_4(x_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return torch.sigmoid(final)
        else:
            return torch.sigmoid(final_4)

    def _preprocess_forward(self, inputs):
        return inputs
