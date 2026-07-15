import pytest
import torch
import torch.nn as nn

from lis2.loss import DiceLoss, BinaryCrossEntropyDiceSum, BaseLoss


@pytest.fixture
def loss_config():
    config = {
        "type": "BinaryCrossEntropyDiceSum",
    }
    return config


def test_dice_basic(loss_config):

    y_pred = torch.tensor([[1.0], [0.0]])
    y_true = torch.tensor([[1.0], [1.0]])
    loss_config["type"] = "DiceLoss"
    loss_config["smooth"] = 1.0
    loss = BaseLoss.from_config(loss_config)
    loss.smooth = 1.0

    out = loss(y_pred, y_true)
    # manual expected: intersection=1, sum_pred=1, sum_true=2 -> dsc=(2*1+1)/(1+2+1)=3/4
    assert pytest.approx(0.25, rel=1e-6) == float(out)


def test_dice_all_zeros(loss_config):
    y_pred = torch.zeros((3, 1))
    y_true = torch.zeros((3, 1))
    loss_config["type"] = "DiceLoss"
    loss_config["smooth"] = 1.0
    loss = BaseLoss.from_config(loss_config)
    loss.smooth = 1.0

    out = loss(y_pred, y_true)
    assert float(out) == pytest.approx(0.0, abs=1e-7)


def test_dice_size_mismatch_raises(loss_config):
    y_pred = torch.zeros((2, 1))
    y_true = torch.zeros((2, 2))

    loss_config["type"] = "DiceLoss"
    loss_config["smooth"] = 1.0
    loss = BaseLoss.from_config(loss_config)
    loss.smooth = 1.0

    with pytest.raises(AssertionError):
        loss(y_pred, y_true)


def test_bce_dice_sum_matches_components(loss_config):
    y_pred = torch.tensor([[0.9], [0.1]])
    y_true = torch.tensor([[1.0], [0.0]])
    loss_config["alpha"] = 0.7
    loss = BaseLoss.from_config(loss_config)
    # set alpha and smooth explicitly
    loss.alpha = 0.7
    loss.dice_loss.smooth = 1.0

    # compute expected using the internal components
    expected = loss.alpha * loss.bce_loss(y_pred, y_true) + (
        1 - loss.alpha
    ) * loss.dice_loss(y_pred, y_true)

    out = loss(y_pred, y_true)
    assert float(out) == pytest.approx(float(expected), rel=1e-6)
