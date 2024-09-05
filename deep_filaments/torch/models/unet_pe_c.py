"""UNet_pe_c implementation.

Code from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/UNet_pe_c.py
"""
from collections import OrderedDict

import torch
from torch import nn

class UNet_pe_c(nn.Module):
    """
    UNet_pe_c_pe implementation with batch normalization.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image. Default is 1.
    out_channels : int
        Number of channels in the model output. Default is 1.
    init_features : int
        Number of output channels of the 1st convolutional block.
        Default is 64.
    """

    _CONV_KERNEL_SIZE = 3
    _CONV_PADDING = 1
    _MAX_POOL_KERNEL_SIZE = 2
    _MAX_POOL_STRIDE = 2
    _UPCONV_KERNEL_SIZE = 2
    _UPCONV_STRIDE = 2

    def __init__(self, in_channels=3, out_channels=1, init_features=64, encoding="sym"):
        super().__init__()
        self.encoding = encoding

        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(
            kernel_size=UNet_pe_c._MAX_POOL_KERNEL_SIZE, stride=UNet_pe_c._MAX_POOL_STRIDE
        )
        self.encoder2 = self._block(features, 2 * features, name="enc2")
        self.pool2 = nn.MaxPool2d(
            kernel_size=UNet_pe_c._MAX_POOL_KERNEL_SIZE, stride=UNet_pe_c._MAX_POOL_STRIDE
        )
        self.encoder3 = self._block(2 * features, 4 * features, name="enc3")
        self.pool3 = nn.MaxPool2d(
            kernel_size=UNet_pe_c._MAX_POOL_KERNEL_SIZE, stride=UNet_pe_c._MAX_POOL_STRIDE
        )
        self.encoder4 = self._block(4 * features, 8 * features, name="enc4")
        self.pool4 = nn.MaxPool2d(
            kernel_size=UNet_pe_c._MAX_POOL_KERNEL_SIZE, stride=UNet_pe_c._MAX_POOL_STRIDE
        )

        self.bottleneck = self._block(8 * features, 16 * features, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            16 * features,
            features * 8,
            kernel_size=UNet_pe_c._UPCONV_KERNEL_SIZE,
            stride=UNet_pe_c._UPCONV_STRIDE,
        )
        self.decoder4 = self._block(2 * 8 * features, 8 * features, name="dc4")
        self.upconv3 = nn.ConvTranspose2d(
            8 * features,
            4 * features,
            kernel_size=UNet_pe_c._UPCONV_KERNEL_SIZE,
            stride=UNet_pe_c._UPCONV_STRIDE,
        )
        self.decoder3 = self._block(2 * 4 * features, 4 * features, name="dc3")
        self.upconv2 = nn.ConvTranspose2d(
            4 * features,
            2 * features,
            kernel_size=UNet_pe_c._UPCONV_KERNEL_SIZE,
            stride=UNet_pe_c._UPCONV_STRIDE,
        )
        self.decoder2 = self._block(2 * 2 * features, 2 * features, name="dc2")
        self.upconv1 = nn.ConvTranspose2d(
            2 * features, 
            features,
            kernel_size=UNet_pe_c._UPCONV_KERNEL_SIZE,
            stride=UNet_pe_c._UPCONV_STRIDE,
        )
        self.decoder1 = self._block(2 * features, features, name="dc1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x, position):
        """Define the computation performed at every call."""
        x = torch.cat((x, position), dim=1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool4(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        # concatenate tensors, this is a skip connection
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=UNet_pe_c._CONV_KERNEL_SIZE,
                            padding=UNet_pe_c._CONV_PADDING,
                            bias=False,
                        ),
                    ),
                    (name + "_norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=UNet_pe_c._CONV_KERNEL_SIZE,
                            padding=UNet_pe_c._CONV_PADDING,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def get_pe(self):
        return self.encoding
    
    def position_encoding(self, positions):
        if self.get_pe() == "sin":
            pe_x = torch.absolute(torch.cos(((57000 - positions[:,1,0]) * 0.00319444444400 + 180) * torch.pi / 360))
            pe_y = torch.absolute(torch.cos(((900 - positions[:,0,0]) * 0.00319444444400) * torch.pi / 360))
        elif self.get_pe() == "lin":
            pe_x = positions[:,0,0] / 114000
            pe_y = positions[:,1,0] / 1800
        elif self.get_pe() == "sym":
            pe_x = torch.absolute((positions[:,1,0] - 57000) / 57000)
            pe_y = torch.absolute((positions[:,0,0] - 900) / 900)
        pe_x = torch.unsqueeze(torch.unsqueeze(pe_x.expand(pe_x.shape[0], 32), dim=2).expand(pe_x.shape[0], 32, 32), dim=1)
        pe_y = torch.unsqueeze(torch.unsqueeze(pe_y.expand(pe_y.shape[0], 32), dim=2).expand(pe_y.shape[0], 32, 32), dim=1)
        return torch.cat((pe_x, pe_y), dim=1)