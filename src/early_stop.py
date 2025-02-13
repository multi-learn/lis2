import abc
from typing import Optional

from configurable import Schema, TypedConfigurable


class BaseEarlyStopping(TypedConfigurable, abc.ABC):
    """
    Abstract base class for early stopping strategies.

    This class defines the interface for early stopping mechanisms, which monitor a metric (e.g., loss)
    during training and stop the process if no improvement is observed after a certain number of epochs.

    **Configuration:**

        - **patience** (int): Number of epochs to wait for improvement before stopping.
        - **min_delta** (float): Minimum change in monitored value to qualify as an improvement.

    Example:
        ```python
        config = {
            'type': 'loss_early_stopping',
            'patience': 5,
            'min_delta': 0.01
        }
        early_stopper = EarlyStopping.from_config(config)

        for epoch in range(num_epochs):
            loss = compute_loss()
            if early_stopper.step(loss):
                print("Early stopping triggered.")
                break
        ```
    """

    config_schema = {
        'patience': Schema(int, default=10),
        'min_delta': Schema(float, default=0.0),
    }

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """
        Initializes the EarlyStopping instance with patience and minimum delta.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta

    @abc.abstractmethod
    def step(self, loss: float) -> bool:
        """
        Checks whether training should stop based on the monitored loss.

        Args:
            loss (float): Current value of the monitored loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the early stopping mechanism to its initial state.
        """
        raise NotImplementedError


class LossEarlyStopping(BaseEarlyStopping):
    """
    Early stopping implementation that monitors training loss.

    Stops training if the loss does not improve by at least `min_delta` for a number of consecutive
    epochs equal to `patience`.

    **Configuration:**
        **counter** (int): Tracks the number of epochs without improvement.
        **best_loss** (Optional[float]): Best observed loss value during training.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        """
        Initializes the LossEarlyStopping instance.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
        """
        super().__init__(patience, min_delta)
        self.counter = 0
        self.best_loss: Optional[float] = None

    def step(self, loss: float) -> bool:
        """
        Updates the early stopping state based on the current loss.

        Args:
            loss (float): Current value of the monitored loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def reset(self) -> None:
        """
        Resets the early stopping state to its initial condition.
        """
        self.counter = 0
        self.best_loss = None
