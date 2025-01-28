"""
Early Stopping Module

This module provides an implementation of early stopping mechanisms for machine learning training processes.
The `EarlyStopping` base class and its implementations allow dynamic monitoring of training loss and termination
of training when improvements are no longer observed.

Classes:
    - EarlyStopping: Abstract base class for defining early stopping strategies.
    - LossEarlyStopping: Monitors training loss and stops training if the loss does not improve after a certain number of epochs.

Usage:
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
"""
import abc

from configs.config import Schema, TypedCustomizable


class EarlyStopping(TypedCustomizable, abc.ABC):
    """
    Abstract base class for early stopping strategies.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in monitored value to qualify as an improvement.
    """

    config_schema = {
        'patience': Schema(int, default=10),
        'min_delta': Schema(float, default=0.0),
    }

    @abc.abstractmethod
    def step(self, loss):
        """
        Checks whether training should stop based on the monitored loss.

        Args:
            loss (float): Current value of the monitored loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """
        Resets the early stopping mechanism to its initial state.
        """
        raise NotImplementedError


class LossEarlyStopping(EarlyStopping):
    """
    Early stopping implementation that monitors training loss.

    Stops training if the loss does not improve by at least `min_delta` for a number of consecutive
    epochs equal to `patience`.

    Attributes:
        counter (int): Tracks the number of epochs without improvement.
        best_loss (float or None): Best observed loss value during training.
    """

    def __init__(self):
        """
        Initializes the LossEarlyStopping instance.
        """
        self.counter = 0
        self.best_loss = None

    def step(self, loss):
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

    def reset(self):
        """
        Resets the early stopping state to its initial condition.
        """
        self.counter = 0
        self.best_loss = None
