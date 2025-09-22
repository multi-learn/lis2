"""
VNet network

Code from https://github.com/mattmacy/vnet.pytorch/blob/master/vnet.py

Clarify by François-Xavier Dupé
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as func
from configurable import Schema

from .base_model import BaseModel


def passthrough(x, **kwargs):
    return x


def elu_cons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm2d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, i_input):
        if i_input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(i_input.dim()))

    def forward(self, i_input):
        self._check_input_dim(i_input)
        return func.batch_norm(
            i_input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = elu_cons(elu, nchan)
        self.conv1 = nn.Conv2d(nchan, nchan, kernel_size=(5, 5), padding=2)
        self.bn1 = ContBatchNorm2d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_n_conv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, out_chans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2)
        self.bn1 = ContBatchNorm2d(16)
        self.relu1 = elu_cons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_chans, n_convs, elu, dropout=False):
        super(DownTransition, self).__init__()
        out_chans = 2 * in_chans
        self.down_conv = nn.Conv2d(
            in_chans, out_chans, kernel_size=(2, 2), stride=(2, 2)
        )
        self.bn1 = ContBatchNorm2d(out_chans)
        self.do1 = passthrough
        self.relu1 = elu_cons(elu, out_chans)
        self.relu2 = elu_cons(elu, out_chans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_n_conv(out_chans, n_convs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, in_chans, out_chans, n_convs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_chans, out_chans // 2, kernel_size=(2, 2), stride=(2, 2)
        )
        self.bn1 = ContBatchNorm2d(out_chans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout2d()
        self.relu1 = elu_cons(elu, out_chans // 2)
        self.relu2 = elu_cons(elu, out_chans)
        if dropout:
            self.do1 = nn.Dropout2d()
        self.ops = _make_n_conv(out_chans, n_convs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_chans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, 2, kernel_size=(5, 5), padding=2)
        self.bn1 = ContBatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=(1, 1))
        self.relu1 = elu_cons(elu, 2)
        if nll:
            self.softmax = torch.log_softmax
        else:
            self.softmax = torch.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(BaseModel):
    """
    VNet model for 2d image segmentation.

    This class implements the VNet architecture, which is designed for 2d image segmentation tasks.
    It includes convolutional layers, batch normalization, and residual connections to improve performance.

    Configuration:
    name (str): The name of the VNet model.
    elu (bool): Whether to use ELU activation function. Default is True.
    nll (bool): Whether to use negative log-likelihood loss. Default is False.
    dropout_rate (float): Dropout rate for the dropout layers. Default is 0.5.
    activation (str): Activation function to use. Default is 'elu'.
    n_convs (list of int): Number of convolutions in each layer. Default is [1, 2, 3, 2, 2].

    Example Configuration (Python):
        .. code-block:: python

            config = {
                "name": "example_vnet",
                "elu": True,
                "nll": False,
                "dropout_rate": 0.5,
                "activation": "elu",
                "n_convs": [1, 2, 3, 2, 2]
            }

    Aliases:

    vnet
    """

    config_schema = {
        "elu": Schema(bool, default=True),
        "nll": Schema(bool, default=False),
        "dropout_rate": Schema(float, default=0.5),
        "activation": Schema(str, default="elu"),
        "n_convs": Schema(list, default=[1, 2, 3, 2, 2]),
    }

    def __init__(self, *args, **kwargs):
        super(VNet, self).__init__(*args, **kwargs)

        self.in_tr = InputTransition(16, self.elu)
        self.down_tr32 = DownTransition(16, self.n_convs[0], self.elu)
        self.down_tr64 = DownTransition(32, self.n_convs[1], self.elu)
        self.down_tr128 = DownTransition(64, self.n_convs[2], self.elu, dropout=True)
        self.down_tr256 = DownTransition(128, self.n_convs[3], self.elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, self.n_convs[4], self.elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, self.n_convs[3], self.elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, self.n_convs[1], self.elu)
        self.up_tr32 = UpTransition(64, 32, self.n_convs[0], self.elu)
        self.out_tr = OutputTransition(32, self.elu, self.nll)

    def core_forward(self, x):
        """
        Core forward pass of the VNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out

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

    def postprocess_forward(self, x: Any) -> torch.Tensor:
        """
        Postprocess the output data after the core forward pass.

        Args:
            x (Any): The output tensor after the forward pass.

        Returns:
            torch.Tensor: The postprocessed output tensor.
        """
        return x
