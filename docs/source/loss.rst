Loss Functions
==============

This module provides a collection of customizable loss functions implemented in PyTorch. Each loss function is designed for different tasks, including segmentation and classification.

.. currentmodule:: loss


Adding a Custom Loss Function
----------------------------

To add a custom loss function, you need to define a new class that extends the ``BaseLoss`` class and implements the ``forward`` method.

Here is an example of how to define a custom loss function class:

.. code-block:: python

    from loss import BaseLoss
    import torch
    import torch.nn as nn

    class CustomLoss(BaseLoss):
        """
        CustomLoss is an example implementation extending BaseLoss.

        This class implements a custom loss function logic.
        """

        def __init__(self):
            super().__init__()

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # Custom loss computation logic
            loss = torch.mean((y_pred - y_true) ** 2)
            return loss

After defining the custom class, you can configure it using a YAML configuration file and update the ``__init__.py`` file to make it accessible.

Conclusion
----------

You have now successfully added a custom loss function by extending the ``BaseLoss`` class and implementing the necessary methods. You can further customize the loss function by modifying the logic in the ``forward`` method according to your needs.


BaseLoss Class
--------------

.. autoclass:: BaseLoss
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseLoss`` class is an abstract base class that provides a structure for defining customizable loss functions. It allows dynamic configuration of loss functions.

**Configuration:**
    - **name** (str): The name of the loss function.



DiceLoss Class
--------------

.. autoclass:: DiceLoss
   :members:
   :undoc-members:
   :show-inheritance:

The ``DiceLoss`` class implements the Dice coefficient loss, which is commonly used in segmentation tasks to measure overlap between predicted and ground truth masks.

**Configuration:**
    - **name** (str): The name of the loss function.
    - **smooth** (float): Smoothing factor to avoid division by zero. Default is 1.0.

Example Configuration (YAML):

.. code-block:: yaml

    name: "dice_loss"
    smooth: 1.0

Aliases:
    - dice_loss

BinaryCrossEntropyDiceSum Class
-------------------------------

.. autoclass:: BinaryCrossEntropyDiceSum
   :members:
   :undoc-members:
   :show-inheritance:

The ``BinaryCrossEntropyDiceSum`` class combines Binary Cross-Entropy (BCE) loss and Dice loss. It is commonly used in segmentation tasks to balance pixel-wise classification with region-based overlap.

**Configuration:**
    - **name** (str): The name of the loss function.
    - **alpha** (float): Weighting factor for BCE loss. Default is 0.5.

Example Configuration (YAML):

.. code-block:: yaml

    name: "bce_dice_loss"
    alpha: 0.5

Aliases:
    - bce_dice_loss
