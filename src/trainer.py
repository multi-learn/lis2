import os
from pathlib import Path
from typing import Union, List, Dict, Optional, Any

import matplotlib
import pandas as pd
import torch
from configurable import Configurable, Schema, Config, GlobalConfig

from src.datasets.dataset import BaseDataset

from src.datasets.filaments_dataset import FilamentsDataset
from src.early_stop import BaseEarlyStopping
from src.metrics import MetricManager
from src.models.base_model import BaseModel
from src.optimizer import BaseOptimizer
from src.scheduler import BaseScheduler
from src.trackers import Trackers
from src.utils.distributed import get_rank_num, is_main_gpu, get_world_size, setup, cleanup, synchronize, reduce_sum

matplotlib.use("Agg")
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from torch import nn
from abc import ABC, abstractmethod


class ITrainer(ABC, Configurable):
    """
    Interface for the Trainer class, defining the abstract method for training.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Abstract method to start the training process.
        """
        pass


class Trainer(ITrainer):
    """
    Trainer class responsible for managing the training, validation, and testing loops.

    The Trainer class is designed to handle the complete lifecycle of model training, including
    configuration setup, data loading, training loops, validation, and testing. It supports
    distributed training, early stopping, and various tracking mechanisms.

    Configuration:
        - **name** (str): The name of the training run.
        - **output_dir** (Union[Path, str]): Directory to save outputs.
        - **model** (Config): Configuration for the model (:class:`BaseModel`).
        - **train_dataset** (Config): Configuration for the training dataset (:class:`BaseDataset`).
        - **val_dataset** (Config): Configuration for the validation dataset (:class:`BaseDataset`).
        - **test_dataset** (Optional[Config]): Configuration for the test dataset (:class:`BaseDataset`). Default is None.
        - **optimizer** (Config): Configuration for the optimizer (:class:`BaseOptimizer`).
        - **scheduler** (Optional[Config]): Configuration for the learning rate scheduler (:class:`BaseScheduler`). Default is None.
        - **early_stopper** (Optional[Union[Config, bool]]): Configuration for early stopping (:class:`BaseEarlyStopping`), work with loss of validation. If True, using :class:`LossEarlyStopping` by default. If False, early stopping is disabled.
        - **batch_size** (int): Batch size for training. Default is 256.
        - **num_workers** (int): Number of workers for data loading. Default is the number of available CPUs.
        - **epochs** (int): Number of training epochs. Default is 100.
        - **save_interval** (int): Interval for saving model checkpoints. Default is 10.
        - **trackers** (Config): Configuration for trackers (:class:`BaseTracker`). Default is an empty dictionary.
        - **save_last** (bool): Whether to save the last model checkpoint. Default is False.
        - **metrics** (List[Config]): List of metrics to track (:class:`BaseMetric`). Default is a list with 'map', 'dice', and 'roc_auc' metrics.

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "example_run"
            output_dir: "/path/to/output"
            model:
                type: "ExampleModel"
                params: {}
            train_dataset:
                type: "ExampleDataset"
                params: {}
            val_dataset:
                type: "ExampleDataset"
                params: {}
            test_dataset:
                type: "ExampleDataset"
                params: {}
            optimizer:
                type: "Adam"
                params:
                    lr: 0.001
            scheduler:
                type: "StepLR"
                params:
                    step_size: 10
            early_stopper:
                type: "EarlyStopping"
                params:
                    patience: 5
            batch_size: 256
            num_workers: 8
            epochs: 100
            save_interval: 10
            trackers: {}
            save_last: False
            metrics:
                - type: "map"
                - type: "dice"
                - type: "roc_auc"

    Aliases:
        epoch
    """

    config_schema = {
        "output_dir": Schema(Path),
        "run_name": Schema(str),
        "model": Schema(type=Config),
        "train_dataset": Schema(type=Config),
        "val_dataset": Schema(type=Config),
        "test_dataset": Schema(type=Config, optional=True),
        "optimizer": Schema(type=Config),
        "scheduler": Schema(type=Config, optional=True),
        "early_stopper": Schema(type=Union[Config, bool], optional=True),
        "batch_size": Schema(int, optional=True, default=256),
        "num_workers": Schema(int, optional=True, default=os.cpu_count()),
        "epochs": Schema(int, optional=True, default=100, aliases=["epoch"]),
        "save_interval": Schema(int, optional=True, default=10),
        "trackers": Schema(type=Config, optional=True, default={}),
        "save_last": Schema(bool, optional=True, default=False),
        "metrics": Schema(
            type=List[Config],
            optional=True,
            default=[
                {"type": "map"},
                {"type": "dice"},
                {"type": "roc_auc"},
            ],
        ),
    }

    def __init__(self, force_device: Optional[str] = None, multi_gpu=False) -> None:
        """
        Initialize the Trainer with configuration and setup.

        Args:
            force_device (Optional[str]): Force the device to be used (e.g., 'cpu', 'cuda').
        """
        super().__init__()
        os.makedirs(self.output_dir / self.run_name, exist_ok=True)
        self.save_dict_to_yaml(
            self.config, self.output_dir / self.run_name / "config.yaml"
        )
        self.setup_device(force_device, multi_gpu)

        self.num_workers = self.num_workers // 2
        self.logger.debug(f"Device: {self.device}")
        self.model = BaseModel.from_config(self.model).to(self.device)
        self.train_dataset = BaseDataset.from_config(self.train_dataset)
        self.train_dataloader = self._create_dataloader(
            self.train_dataset, is_train=True
        )
        self.val_dataset = BaseDataset.from_config(self.val_dataset)
        self.val_dataloader = self._create_dataloader(self.val_dataset, is_train=True)

        if self.test_dataset is not None:
            self.test_dataset = BaseDataset.from_config(self.test_dataset)
            self.test_dataloader = self._create_dataloader(
                self.test_dataset, is_train=False
            )

        self.optimizer = BaseOptimizer.from_config(
            self.optimizer.copy(), params=self.model.parameters()
        )
        self.scheduler = (
            BaseScheduler.from_config(self.scheduler.copy(), optimizer=self.optimizer)
            if self.scheduler
            else None
        )
        self.early_stopper = (
            BaseEarlyStopping.from_config(
                self.early_stopper
                if self.early_stopper is not None
                else {"Type": "LossEarlyStopping"}
            )
            if self.early_stopper
            else None
        )

        self.tracker = Trackers(self.trackers, self.output_dir / self.run_name)
        self.metrics_fn = MetricManager(self.metrics)
        self.epochs_run = 0
        self.best_loss = float("inf")

        if torch.cuda.device_count() >= 2:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu_id], output_device=self.gpu_id
            )
            self.model = self.model.module
            self.logger.debug(
                f"Model wrapped with DistributedDataParallel on GPU {self.gpu_id}"
            )
            synchronize()
        self.loss_fn = torch.nn.BCELoss()

    def setup_device(self, force_device=None, multi_gpu=False):
        """
        Sets up the computing device, handling GPU assignments in a distributed setting.

        Args:
            force_device (str or None): Manually specified device (e.g., "cuda:0", "cpu").
                                        If None, it assigns the appropriate device based on availability.
        """
        world_size = get_world_size()
        self.gpu_id = get_rank_num() if torch.cuda.device_count() >= 2 else 0
        if force_device is not None:
            self.device = torch.device(force_device)
        else:
            self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        if multi_gpu:
            self.logger.debug(f"Setting up distributed training with {world_size} GPUs")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.gpu_id)

    def preconditions(self) -> None:
        """
        Check preconditions before starting the training process.
        """
        assert self.epochs > 0, "Number of epochs must be greater than 0"
        assert self.batch_size > 0, "Batch size must be greater than 0"
        assert self.num_workers > 0, "Number of workers must be greater than 0"
        self.logger.debug("Preconditions passed")

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Start the training process, including validation and test phases.

        Returns:
            Dict[str, Any]: Final training information including metrics and model paths.
        """
        if is_main_gpu():
            self.tracker.init()
            self.logger.debug("Tracker initialized")

        loop = tqdm(
            range(self.epochs_run, self.epochs),
            desc="Training",
            unit="epoch...",
            disable=not is_main_gpu(),
        )
        self.model.train()
        for epoch in loop:
            train_loss = self._run_loop_train(epoch)
            val_loss, _ = (
                self._run_loop_test(self.val_dataloader, description=f"Epoch {epoch}/{self.epochs} - Validation")
                if hasattr(self, "val_dataloader")
                else None
            )
            if is_main_gpu():
                lr = self.optimizer.param_groups[0]["lr"]
                log = {"train_loss": train_loss.item()}
                if val_loss is not None:
                    log["val_loss"] = val_loss.item()
                log["lr"] = lr
                if hasattr(self, "val_dataloader"):
                    log.update(self.metrics_fn.to_dict())
                self.tracker.log(epoch, log)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_snapshot(
                        epoch,
                        self.output_dir / self.run_name / "best.pt",
                        train_loss,
                    )

                if epoch % self.save_interval == 0 and epoch > 0:
                    self._save_snapshot(
                        epoch,
                        self.output_dir / self.run_name / f"save_{epoch}.pt",
                        train_loss,
                    )
                if self.save_last:
                    self._save_snapshot(
                        epoch,
                        self.output_dir / self.run_name / "last.pt",
                        train_loss,
                    )

                loop.set_postfix_str(
                    f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {lr:.6f}"
                    if val_loss is not None
                    else f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | LR: {lr:.6f}"
                )

            if self.early_stopper and self.early_stopper.step(val_loss):
                if is_main_gpu():
                    self.logger.info(
                        f"Early stopping at epoch {epoch}. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}"
                    )
                break
        synchronize()
        if is_main_gpu():
            self.tracker.finish()
            self.logger.info(
                f"Training finished. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}, "
                f"saved at {self.output_dir / self.run_name / 'best.pt'}"
            )
        else: 
            self.logger.debug(
                f"Training finished.")
        final_info = self.get_final_info()
        synchronize()
        return final_info

    def _run_loop_train(self, epoch: int) -> torch.Tensor:
        """
        Run the training loop for a given epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            torch.Tensor: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        averaging_coef = 0
        iters = len(self.train_dataloader)

        loop = tqdm(
            enumerate(self.train_dataloader),
            total=iters,
            desc=f"Epoch {epoch}/{self.epochs} - Training",
            unit="batch",
            disable=not is_main_gpu(),
            leave=False,
        )

        for i, batch in loop:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, idx = self._run_batch(batch, update_params=True, update_metrics=False)
            total_loss += loss.detach() * idx.sum()
            averaging_coef += idx.sum()

            if is_main_gpu():
                loop.set_postfix_str(f"Train Loss: {total_loss / averaging_coef:.6f}")

        avg_loss = total_loss / averaging_coef
        self.scheduler.step()
        return avg_loss

    def _run_loop_test(
            self,
            dataloader: Optional[DataLoader] = None,
            description: str = "Validation"
    ) -> torch.Tensor:
        """
        Runs the validation or test loop.

        Args:
            dataloader (Optional[DataLoader]): Dataloader for validation or test.
            description (str): Description for the progress bar.
            update_metrics (bool, optional): If True, updates metrics.

        Returns:
            torch.Tensor: Average loss.
        """
        dataloader = dataloader if dataloader is not None else self.val_dataloader
        self.model.eval()
        total_loss = 0
        iters = len(dataloader)
        results = []
        with torch.no_grad():
            loop = tqdm(
                enumerate(dataloader),
                total=iters,
                desc=description,
                unit="batch",
                disable=not is_main_gpu(),
                leave=False,
            )
            for i, batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss, _ = self._run_batch(batch, update_params=False, update_metrics=True)
                total_loss += loss
                if is_main_gpu():
                    loop.set_postfix_str(f"{description} Loss: {total_loss / (i + 1):.6f}")

                results.append(
                    {"batch": i, "loss": loss.item(), **self.metrics_fn.to_dict()}
                )

        avg_loss = total_loss / iters
        reduce_sum(avg_loss)
        return avg_loss, results

    def _run_batch(
            self, batch: Dict[str, torch.Tensor], update_params: bool = True, update_metrics: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a training batch with masking, metric updates, and optional optimization.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            update_params (bool, optional): If True, updates model parameters.
            update_metrics (bool, optional): If True, updates metrics.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Loss and sum of indices for normalization.
        """
        preds = self.model(**batch)
        target = batch["target"]

        if "labelled" in batch:
            idx = batch["labelled"] == 1
        else:
            idx = torch.ones_like(target, dtype=torch.bool)

        temp_loss = self.loss_fn(preds[idx], target[idx])

        if update_params:
            self.optimizer.zero_grad()
            temp_loss.backward()
            self.optimizer.step()

        if update_metrics:
            self.metrics_fn.update(
                torch.flatten(preds[idx]).detach().cpu().numpy(),
                torch.flatten(target[idx]).detach().cpu().numpy(),
                torch.flatten(idx).detach().cpu().numpy(),
            )

        idx_sum = torch.flatten(idx).detach().cpu().numpy().sum()
        temp_loss = temp_loss.detach().cpu()

        return temp_loss, idx_sum

    def _save_snapshot(self, epoch: int, path: str, loss: torch.Tensor) -> None:
        """
        Save a snapshot of the training progress.

        Args:
            epoch (int): Current epoch number.
            path (str): Path to save the snapshot.
            loss (torch.Tensor): Current loss value.
        """
        snapshot = {
            "MODEL": {
                "MODEL_CONFIG": self.config["model"],
                "MODEL_STATE": self.model.state_dict(),
            },
            "TRAIN_INFO": {
                "EPOCHS_RUN": epoch,
                "BEST_LOSS": loss,
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "SCHEDULER_STATE": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            },
            "GLOBAL_CONFIG": self.global_config.to_dict(),
        }
        torch.save(snapshot, path)
        self.logger.debug(
            f"Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}"
        )

    def _create_dataloader(
        self, dataset: BaseDataset, is_train: bool = True
    ) -> DataLoader:
        """
        Create a dataloader for the given dataset.

        Args:
            dataset (BaseDataset): Dataset to create a dataloader for.
            is_train (bool): Whether the dataloader is for training.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        sampler = (
            DistributedSampler(
                dataset, rank=get_rank_num(), shuffle=is_train, drop_last=False
            )
            if torch.cuda.device_count() >= 2 and is_train
            else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            persistent_workers=True,
            shuffle=is_train if sampler is None else False,
            num_workers=self.num_workers,
            sampler=sampler,
            
        )

        return dataloader

    @classmethod
    def from_snapshot(cls, snapshot_path: str) -> "Trainer":
        """
        Load a snapshot of the training progress.

        Args:
            snapshot_path (str): Path to the snapshot file.

        Returns:
            Trainer: Trainer instance loaded from the snapshot.
        """
        snapshot = torch.load(snapshot_path)
        global_config = GlobalConfig(config=snapshot["GLOBAL_CONFIG"])
        trainer = cls.from_config(global_config.to_dict())
        trainer.model.load_state_dict(snapshot["MODEL"]["MODEL_STATE"])
        trainer.optimizer.load_state_dict(snapshot["TRAIN_INFO"]["OPTIMIZER_STATE"])
        if snapshot["TRAIN_INFO"]["SCHEDULER_STATE"] is not None:
            trainer.scheduler.load_state_dict(snapshot["TRAIN_INFO"]["SCHEDULER_STATE"])
        trainer.epochs_run = snapshot["TRAIN_INFO"]["EPOCHS_RUN"]
        trainer.best_loss = snapshot["TRAIN_INFO"]["BEST_LOSS"]
        trainer.logger.info(
            f"Snapshot loaded from {snapshot_path} | Epochs run: {trainer.epochs_run} | Best loss: {trainer.best_loss}"
        )
        return trainer

    def run_final_test(
        self,
        csv_path: str = "test_results.csv",
    ) -> Dict[str, Any]:
        """
        Execute the test phase on a specified test dataset and save results to a CSV file.

        Args:
            test_dataset (Optional[BaseDataset]): Dataset object for testing. If None, uses `self.test_dataloader`.
            csv_path (str): Path to save the test results as a CSV file.

        Returns:
            Dict[str, Any]: Dictionary containing the average loss and computed metrics.
        """
        if hasattr(self, "test_dataset"):
            self.test_dataloader = self._create_dataloader(self.test_dataset, is_train=False)
        elif hasattr(self, "val_dataset"):
            self.test_dataloader = self._create_dataloader(self.val_dataset, is_train=False)
        else:
            raise ValueError("Neither test_dataset nor validation_dataloader is provided.")
        assert hasattr(self, "test_dataloader"), "Test or validation dataset not provided."
        self.model.eval()
        self.metrics_fn.reset()
        avg_loss, results =  self._run_loop_test(self.test_dataloader, description=f"Test")
        df = pd.DataFrame(results)
        csv_path = self.output_dir / self.run_name / csv_path
        df.to_csv(csv_path, index=False)
        final_metrics = {"avg_loss": avg_loss, **self.metrics_fn.to_dict()}
        metrics_str = "\n".join([f"  {metric.replace('_', ' ').capitalize():<20}: {value:.6f}" for metric, value in final_metrics.items()])
        if is_main_gpu():
            self.logger.info(f"Test results saved to {csv_path}\nAverage Loss: {avg_loss:.6f}\n"
                     f"Final Metrics:\n{metrics_str}")
        return final_metrics

    def get_final_info(self) -> Dict[str, Any]:
        """
        Return the final training information including metrics, best model path, and other relevant details.
        """
        final_info = {
            "best_loss": self.best_loss,
            "epochs_run": self.epochs,
            "run_name": self.run_name,
            "output_dir": self.output_dir,
            "final_model_path": self.output_dir / self.run_name / "best.pt",
            "last_model_path": self.output_dir / self.run_name / "last.pt",
            "metrics_train": self.metrics_fn.to_dict(),
        }

        if hasattr(self, "test_dataloader"):
            model = BaseModel.from_snapshot(self.output_dir / self.run_name / 'best.pt')
            self.model = model.to(self.device)
            final_metrics = self.run_final_test()
            final_info["metrics_test"] = final_metrics
        final_info = recursive_to_cpu(final_info)
        return final_info

def recursive_to_cpu(data):
    """
    Convert all CUDA tensors in the data structure to CPU, detach them to remove
    computation graph dependencies, and ensure they are serializable.
    """
    if torch.is_tensor(data):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: recursive_to_cpu(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [recursive_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_cpu(item) for item in data)
    return data