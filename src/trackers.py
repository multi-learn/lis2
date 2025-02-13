import csv
import os
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import mlflow
import wandb
from configurable import Schema, TypedConfigurable


class BaseTracker(ABC, TypedConfigurable):
    """
    Abstract base class for all trackers.

    This class provides a common interface for logging metrics during model training and evaluation.
    To create a new tracker, subclass `BaseTracker` and implement the `init`, `log`, and optionally `close` methods.

    Attributes:
        **output_run** (str): Path to the directory where logs are stored.

    Example:
        ```python
        class MyCustomTracker(BaseTracker):
            def init(self):
                # Initialize custom tracker
                pass

            def log(self, epoch, log_dict):
                # Log metrics using custom logic
                pass

            def close(self):
                # Clean up resources if necessary
                pass

        tracker = MyCustomTracker(output_run='./logs')
        tracker.init()
        tracker.log(epoch=1, log_dict={'loss': 0.1, 'accuracy': 0.9})
        tracker.close()
        ```
    """

    def __init__(self, *args, output_run: str = '.', **kwargs):
        super().__init__(*args, **kwargs)
        self.output_run = output_run

    @abstractmethod
    def init(self) -> None:
        """
        Initializes the tracker. Should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def log(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """
        Logs metrics or parameters for the given epoch.

        Args:
            epoch (int): The current epoch number.
            log_dict (Dict[str, float]): A dictionary of metrics to log.
        """
        pass

    def close(self) -> None:
        """
        Closes the tracker and performs cleanup if needed.
        Can be optionally overridden by subclasses.
        """
        pass

class Wandb(BaseTracker):
    """
    Tracker using Weights and Biases (Wandb) for logging.

    Attributes:
        **entity** (str): Wandb entity.
    """

    config_schema = {
        'entity': Schema(str),
    }

    def init(self, config: Optional[Dict[str, any]] = None) -> None:
        """
        Initializes the Wandb tracker with the provided configuration.
        """
        timestamp = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        os.environ["WANDB_CACHE_DIR"] = os.path.join(self.output_run, "WANDB_cache")
        os.environ["WANDB_DIR"] = os.path.join(self.output_run, "WANDB")
        self.run = wandb.init(
            project=self.config.wandb_config.project,
            resume="auto" if self.config.wandb_config.resume else None,
            mode=os.environ.get("WANDB_MODE", "online"),
            entity=self.config.wandb_config.entity,
            name=f"{self.config.run_name}_{timestamp}",
            config=self.config.to_dict(),
        )

    def log(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """
        Logs metrics to Wandb.

        Args:
            epoch (int): The current epoch number.
            log_dict (Dict[str, float]): A dictionary of metrics to log.
        """
        wandb.log(log_dict, step=epoch)

    def close(self) -> None:
        """
        Closes the Wandb tracker.
        """
        wandb.finish()

class Mlflow(BaseTracker):
    """
    Tracker using MLflow for logging.

    Attributes:
        **tracking_uri** (str): URI for the MLflow server.
        **experiment_name** (str): Name of the MLflow experiment.
    """

    def init(self) -> None:
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

    def log(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """
        Logs metrics to MLflow.

        Args:
            **epoch** (int): The current epoch number.
            **log_dict** (Dict[str, float]): A dictionary of metrics to log.
        """
        mlflow.log_metrics(log_dict, step=epoch)

    def close(self) -> None:
        """
        Ends the current MLflow run.
        """
        mlflow.end_run()

class CsvLogger(BaseTracker):
    """
    Tracker for logging metrics to a CSV file.

    Attributes:
        **csv_filename** (str): Path to the CSV file where logs are saved.
        **file_initialized** (bool): Indicates whether the CSV file is initialized.
        **fieldnames** (List[str]): List of field names for the CSV file.
    """

    def __init__(self, output_run: str):
        super().__init__(output_run=output_run)
        self.csv_filename = os.path.join(output_run, "logs_train.csv")
        self.file_initialized = False
        self.fieldnames: Optional[List[str]] = None

    def init(self) -> None:
        """
        Initializes the CSV logger. This method does nothing for CSV logging.
        """
        pass

    def _init_file(self, fieldnames: List[str]) -> None:
        """
        Initializes the CSV file with headers.

        Args:
            fieldnames (List[str]): List of column names for the CSV file.
        """
        os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
        with open(self.csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        self.file_initialized = True

    def log(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """
        Logs metrics to the CSV file.

        Args:
            epoch (int): The current epoch number.
            log_dict (Dict[str, float]): A dictionary of metrics to log.
        """
        current_fieldnames = ["epoch"] + list(log_dict.keys())

        if not self.file_initialized or self.fieldnames != current_fieldnames:
            self.fieldnames = current_fieldnames
            self._init_file(self.fieldnames)

        with open(self.csv_filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({**{"epoch": epoch}, **log_dict})

    def close(self) -> None:
        """
        Finalizes the CSV logger by resetting initialization flags.
        """
        self.file_initialized = False

class Trackers:
    """
    Manages multiple trackers and coordinates logging.

    Attributes:
        **loggers** (List[BaseTracker]): List of initialized tracker instances.
        **loggers_configs** (List[Dict[str, any]]): Configuration for the trackers.
        **output_run** (str): Path to the directory where logs are stored.
        **is_init** (bool): Indicates whether trackers are initialized.
    """

    def __init__(self, loggers_configs: List[Dict[str, any]], output_run: str):
        self.loggers: List[BaseTracker] = []
        self.loggers_configs = loggers_configs
        self.output_run = output_run
        self.is_init = False

    def init(self) -> None:
        """
        Initializes all trackers based on the provided configurations.
        """
        for logger_config in self.loggers_configs:
            logger_cls = BaseTracker.from_config(logger_config, output_run=self.output_run)
            logger_cls.init()
            self.loggers.append(logger_cls)
        self.loggers.append(CsvLogger(self.output_run))
        self.is_init = True

    def log(self, epoch: int, log_dict: Dict[str, float]) -> None:
        """
        Logs metrics to all initialized trackers.

        Args:
            epoch (int): The current epoch number.
            log_dict (Dict[str, float]): A dictionary of metrics to log.
        """
        if not self.is_init:
            raise ValueError("Trackers are not initialized")
        for logger in self.loggers:
            logger.log(epoch, log_dict)

    def finish(self) -> None:
        """
        Finalizes all trackers and performs cleanup.
        """
        if not self.is_init:
            raise ValueError("Trackers are not initialized")
        for logger in self.loggers:
            logger.close()
        self.is_init = False
