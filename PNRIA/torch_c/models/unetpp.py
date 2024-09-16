"""
UNetPlusPlus network
Code from https://github.com/ZJUGiveLab/UNet-Version (no licence)

Clarification and improvement by François-Xavier Dupé
"""
import torch
import torch.nn as nn
from deep_filaments.torch.third_party.layers import unetConv2, unetUp_origin
from deep_filaments.torch.third_party.init_weights import init_weights


class UNetPP(nn.Module):
    """
    UNetPlusPlus model

    Attributes
    ----------
    is_deconv: bool, optional
        True to use deconvolution layers for the upsampling part (decoder)
    in_channel: int, optional
        The number of channels
    is_batchnorm: bool, optional
        True to use batch normalization (recommended)
    is_ds: bool, optional
        [TO DEFINE]
    feature_scale: int, optional
        [TO DEFINE, NOT USED]
    """

    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
        is_ds=True,
    ):
        """
        Constructor

        Parameters
        ----------
        in_channels: int, optional
            The number of channels
        n_classes: int, optional
            The number of classes
        feature_scale: int, optional
            [TO DEFINE, NOT USED]
        is_deconv: bool, optional
            True to use deconvolution layers for the upsampling part (decoder)
        is_batchnorm: bool, optional
            True to use batch normalization (recommended)
        is_ds: bool, optional
            [TO DEFINE]
        """
        super(UNetPP, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        # filters = [32, 64, 128, 256, 512]
        filters = [64, 128, 256, 512, 1024]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp_origin(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp_origin(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp_origin(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp_origin(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp_origin(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp_origin(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp_origin(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp_origin(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp_origin(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp_origin(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, (1, 1))
        self.final_2 = nn.Conv2d(filters[0], n_classes, (1, 1))
        self.final_3 = nn.Conv2d(filters[0], n_classes, (1, 1))
        self.final_4 = nn.Conv2d(filters[0], n_classes, (1, 1))

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        """
        Forward operotion

        Parameters
        ----------
        inputs: torch.Tensor
            The input patch

        Returns
        -------
        The output from the network model
        """
        # column : 0
        x_00 = self.conv00(inputs)
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
