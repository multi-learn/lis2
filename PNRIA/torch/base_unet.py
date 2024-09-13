from collections import OrderedDict
import torch
from torch import nn
from PNRIA.torch.custom_model import BaseModel


class BaseUNet(BaseModel):
    """
    A base class for UNet implementations with configurable number of blocks.
    """

    required_keys = ["in_channels", "out_channels", "features", "num_blocks"]

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

        # Create encoder layers
        for i in range(self.self.num_blocks):
            if i == 0:
                self.encoders.append(self._block(self.in_channels, self.features, f"enc{i+1}"))
            else:
                self.encoders.append(self._block(self.features, self.features * 2, f"enc{i+1}"))
                self.features *= 2
            self.pools.append(nn.MaxPool2d(kernel_size=self._MAX_POOL_KERNEL_SIZE, stride=self._MAX_POOL_STRIDE))

        # Bottleneck
        self.bottleneck = self._block(self.features, self.features * 2, "bottleneck")

        # Create decoder layers
        for i in range(self.num_blocks, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=self._UPCONV_KERNEL_SIZE, stride=self._UPCONV_STRIDE))
            self.decoders.append(self._block(self.features * 2, self.features, f"dec{i}"))
            self.features //= 2

        # Final convolution layer
        self.conv = nn.Conv2d(self.features, self.out_channels, kernel_size=1)

    def forward(self, x):
        enc_outputs = []

        # Encoder forward pass
        for i in range(self.self.num_blocks):
            x = self.encoders[i](x)
            enc_outputs.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward pass with skip connections
        for i in range(self.self.num_blocks):
            x = self.upconvs[i](x)
            x = torch.cat((x, enc_outputs[self.self.num_blocks - i - 1]), dim=1)  # Skip connection
            x = self.decoders[i](x)

        # Final output
        return torch.sigmoid(self.conv(x))

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=self._CONV_KERNEL_SIZE,
                            padding=self._CONV_PADDING,
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
                            kernel_size=self._CONV_KERNEL_SIZE,
                            padding=self._CONV_PADDING,
                            bias=False,
                        ),
                    ),
                    (name + "_norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
