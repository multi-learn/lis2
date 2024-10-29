"""Custom loss functions."""
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice score-based loss."""

    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        """Estimate dice score and convert it to a loss function."""
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
                y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc


class BinaryCrossEntropyDiceSum(nn.Module):
    """A weighted sum of the binary cross-entropy and dice losses."""

    def __init__(self):
        super().__init__()
        # mixing parameter
        self.alpha = 0.5
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_pred, y_true):
        """Estimate weighted sum of BCE and Dice."""
        return self.alpha * self.bce_loss(y_pred, y_true) + (
                1 - self.alpha
        ) * self.dice_loss(y_pred, y_true)
