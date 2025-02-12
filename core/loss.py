import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """
    Dice score-based loss function.

    This loss function is based on the Dice coefficient, which measures the overlap between two samples.
    It is commonly used in image segmentation tasks.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        """
        Initializes the DiceLoss with a smoothing factor.

        Args:
            smooth (float): Smoothing factor to avoid division by zero. Default is 1.0.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Estimates the Dice loss.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: The computed Dice loss.
        """
        assert y_pred.size() == y_true.size(), "Predicted and true tensors must have the same size"

        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)

        return 1.0 - dsc

class BinaryCrossEntropyDiceSum(nn.Module):
    """
    A weighted sum of the binary cross-entropy and Dice losses.

    This loss function combines the binary cross-entropy loss and the Dice loss, allowing for a balance
    between pixel-wise classification and overlap-based segmentation.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """
        Initializes the BinaryCrossEntropyDiceSum with a mixing parameter.

        Args:
            alpha (float): Weighting factor for the binary cross-entropy loss. Default is 0.5.
        """
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Estimates the weighted sum of BCE and Dice losses.

        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): True values.

        Returns:
            torch.Tensor: The computed weighted sum of losses.
        """
        bce = self.bce_loss(y_pred, y_true)
        dice = self.dice_loss(y_pred, y_true)
        return self.alpha * bce + (1 - self.alpha) * dice
