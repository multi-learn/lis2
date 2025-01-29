"""
Tracking System Module

This module provides a framework for tracking and logging metrics during model training and evaluation.
It supports multiple tracking backends, including `Wandb`, `Mlflow`, and `CsvLogger`.
The `Trackers` class acts as a unified interface to manage multiple logging mechanisms.

Classes:
    - BaseTracker: Abstract base class for all trackers.
    - Wandb: Tracker using Weights and Biases.
    - Mlflow: Tracker using MLflow.
    - CsvLogger: Tracker for logging to CSV files.
    - Trackers: Manages multiple trackers and coordinates logging.

Usage:
```python
    trackers = Trackers(
        loggers_configs=[
            {'type': 'wandb', 'entity': 'my_entity'},
            {'type': 'mlflow'}
        ],
        output_run='./logs'
    )
    trackers.init()

    for epoch in range(num_epochs):
        log_dict = {"loss": 0.5, "accuracy": 0.8}
        trackers.log(epoch, log_dict)

    trackers.finish()
```
"""
import csv
import os
import time
from abc import ABC, abstractmethod

import mlflow
import wandb

from configs.config import Schema, TypedCustomizable


class BaseTracker(ABC, TypedCustomizable):
    """
    Abstract base class for all trackers.

    Attributes:
        output_run (str): Path to the directory where logs are stored.
    """

    def __init__(self, *args, output_run='.', **kwargs):
        self.output_run = output_run

    @abstractmethod
    def init(self):
        """
        Initializes the tracker. Should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def log(self, epoch, log_dict):
        """
        Logs metrics or parameters for the given epoch.

        Args:
            epoch (int): The current epoch number.
            log_dict (dict): A dictionary of metrics to log.
        """
        pass

    def close(self):
        """
        Closes the tracker and performs cleanup if needed.
        Can be optionally overridden by subclasses.
        """
        pass


class Wandb(BaseTracker):
    """
    Tracker using Weights and Biases (Wandb) for logging.

    Attributes:
        config_schema (dict): Configuration schema for validating Wandb parameters.
    """

    config_schema = {
        'entity': Schema(str),
    }

    def init(self, config=None):
        """
        Initializes the Wandb tracker with the provided configuration.
        """
        t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        os.environ["WANDB_CACHE_DIR"] = os.path.join(self.output_run, "WANDB_cache")
        os.environ["WANDB_DIR"] = os.path.join(self.output_run, "WANDB")
        self.run = wandb.init(
            project=self.config.wandb_config.project,
            resume="auto" if self.config.wandb_config.resume else None,
            mode=os.environ.get("WANDB_MODE", "online"),
            entity=self.config.wandb_config.entity,
            name=f"{self.config.run_name}_{t}/",
            config=self.config.to_dict(),
        )

    def log(self, epoch, log_dict):
        """
        Logs metrics to Wandb.

        Args:
            epoch (int): The current epoch number.
            log_dict (dict): A dictionary of metrics to log.
        """
        wandb.log(log_dict, step=epoch)

    def close(self):
        """
        Closes the Wandb tracker.
        """
        wandb.finish()


class Mlflow(BaseTracker):
    """
    Tracker using MLflow for logging.

    Attributes:
        tracking_uri (str): URI for the MLflow server.
        experiment_name (str): Name of the MLflow experiment.
    """

    def init(self):
        """
        Initializes the MLflow tracker and starts a new run.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment_name = self.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
        self.run = mlflow.start_run(nested=True, run_name=self.global_config.run_name)
        mlflow.log_params(self.global_config.to_dict())

    def log(self, epoch, log_dict):
        """
        Logs metrics to MLflow.

        Args:
            epoch (int): The current epoch number.
            log_dict (dict): A dictionary of metrics to log.
        """
        mlflow.log_metrics(log_dict, step=epoch)

    def close(self):
        """
        Ends the current MLflow run.
        """
        mlflow.end_run()


class CsvLogger(BaseTracker):
    """
    Tracker for logging metrics to a CSV file.

    Attributes:
        csv_filename (str): Path to the CSV file where logs are saved.
        file_initialized (bool): Indicates whether the CSV file is initialized.
        fieldnames (list): List of field names for the CSV file.
    """

    def __init__(self, output_run):
        self.csv_filename = os.path.join(output_run, "logs_train.csv")
        self.file_initialized = False
        self.fieldnames = None

    def init(self):
        """
        Initializes the CSV logger. This method does nothing for CSV logging.
        """
        pass

    def _init_file(self, fieldnames):
        """
        Initializes the CSV file with headers.

        Args:
            fieldnames (list): List of column names for the CSV file.
        """
        os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
        with open(self.csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        self.file_initialized = True

    def log(self, epoch, log_dict):
        """
        Logs metrics to the CSV file.

        Args:
            epoch (int): The current epoch number.
            log_dict (dict): A dictionary of metrics to log.
        """
        current_fieldnames = ["epoch"] + list(log_dict.keys())

        if not self.file_initialized or self.fieldnames != current_fieldnames:
            self.fieldnames = current_fieldnames
            self._init_file(self.fieldnames)

        with open(self.csv_filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({**{"epoch": epoch}, **log_dict})

    def close(self):
        """
        Finalizes the CSV logger by resetting initialization flags.
        """
        self.file_initialized = False


class Trackers:
    """
    Manages multiple trackers and coordinates logging.

    Attributes:
        loggers (list): List of initialized tracker instances.
        loggers_configs (list): Configuration for the trackers.
        output_run (str): Path to the directory where logs are stored.
        is_init (bool): Indicates whether trackers are initialized.
    """

    def __init__(self, loggers_configs, output_run):
        self.loggers = []
        self.loggers_configs = loggers_configs
        self.output_run = output_run
        self.is_init = False

    def init(self):
        """
        Initializes all trackers based on the provided configurations.
        """
        for logger in self.loggers_configs:
            logger_cls = BaseTracker.from_config(logger, output_run=self.output_run)
            logger_cls.init()
            self.loggers.append(logger_cls)
        self.loggers.append(CsvLogger(self.output_run))
        self.is_init = True

    def log(self, epoch, log_dict):
        """
        Logs metrics to all initialized trackers.

        Args:
            epoch (int): The current epoch number.
            log_dict (dict): A dictionary of metrics to log.
        """
        if not self.is_init:
            raise ValueError("Trackers are not initialized")
        for logger in self.loggers:
            logger.log(epoch, log_dict)

    def finish(self):
        """
        Finalizes all trackers and performs cleanup.
        """
        if not self.is_init:
            raise ValueError("Trackers are not initialized")
        for logger in self.loggers:
            logger.close()
        self.is_init = False
