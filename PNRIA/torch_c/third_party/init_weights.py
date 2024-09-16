"""
Code from https://github.com/ZJUGiveLab/UNet-Version

Clarification by François-Xavier Dupé
"""
import torch
import torch.nn as nn
from torch.nn import init


def weights_init_normal(m):
    """
    Normal initialization of the network weights

    Parameters
    ----------
    m: torch.nn.Module
        The neural network model
    """

    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    """
    Xavier's initialization of the network weights

    Parameters
    ----------
    m: torch.nn.Module
        The neural network model
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    """
    Kaiming's initialization of the network weights

    Parameters
    ----------
    m: torch.nn.Module
        The neural network model
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    """
    Orthogonal initialization of the network weights

    Parameters
    ----------
    m: torch.nn.Module
        The neural network model
    """

    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal"):
    """
    Weight initialization method

    Parameters
    ----------
    net: torch.nn.Module
        The neural network
    init_type: str
        The kind of initialization
    """

    # print('initialization method [%s]' % init_type)
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [%s] is not implemented" % init_type
        )
