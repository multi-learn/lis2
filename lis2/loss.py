import abc

import torch
import torch.nn as nn
from configurable import Schema, TypedConfigurable


class BaseLoss(TypedConfigurable, nn.Module, abc.ABC):
    """
    BaseLoss for defining customizable loss functions.

    This abstract class provides a structure for loss functions that can be configured dynamically.

    Configuration:
        - **name** (str): The name of the loss function.
    """

    @abc.abstractmethod
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between predictions and ground truth.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The ground truth tensor.

        Returns:
            torch.Tensor: Computed loss value.
        """
        pass


class DiceLoss(BaseLoss):
    """
    DiceLoss for segmentation tasks.

    This loss function calculates the Dice coefficient loss, which is commonly used in segmentation
    tasks to measure overlap between the predicted and ground truth masks.

    Configuration:
        - **name** (str): The name of the loss function.
        - **smooth** (float): Smoothing factor to avoid division by zero. Default is 1.0.

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "dice_loss"
            smooth: 1.0

    Aliases:
        dice_loss
    """

    aliases = ["dice_loss"]
    config_schema = {"smooth": Schema(float, default=1.0)}

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert (
            y_pred.size() == y_true.size()
        ), "Predicted and ground truth tensors must have the same size"

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )

        return 1.0 - dsc


class BinaryCrossEntropyDiceSum(BaseLoss):
    """
    BinaryCrossEntropyDiceSum combines BCE and Dice loss.

    This loss function is a weighted sum of Binary Cross-Entropy (BCE) loss and Dice loss, commonly used
    in segmentation tasks to balance pixel-wise classification with region-based overlap.

    Configuration:
        - **name** (str): The name of the loss function.
        - **alpha** (float): Weighting factor for BCE loss. Default is 0.5.

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "bce_dice_loss"
            alpha: 0.5

    Aliases:
        bce_dice_loss
    """

    aliases = ["bce_dice_loss"]
    config_schema = {"alpha": Schema(float, default=0.5)}

    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        return self.alpha * bce + (1 - self.alpha) * dice
