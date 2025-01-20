import logging
import os
from typing import Union
from pathlib import Path

import matplotlib
import torch

from PNRIA.configs.config import Customizable, Schema, Config, GlobalConfig
from PNRIA.torch_c.early_stop import EarlyStopping
from PNRIA.torch_c.loss import BinaryCrossEntropyDiceSum
from PNRIA.torch_c.metrics import Metrics
from PNRIA.torch_c.optim import BaseOptimizer
from PNRIA.torch_c.scheduler import BaseScheduler
from PNRIA.torch_c.trackers import Trackers
from PNRIA.utils.distributed import get_rank, get_rank_num, is_main_gpu

matplotlib.use("Agg")
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from torch import nn

from abc import ABC, abstractmethod


class ITrainer(ABC, Customizable):

    @abstractmethod
    def train(self) -> None:
        pass


class Trainer(ITrainer):
    config_schema = {
        "output_dir": Schema(Union[Path, str]),
        "run_name": Schema(str),
        "optimizer": Schema(type=Config),
        "scheduler": Schema(type=Config, optional=True),
        "early_stopper": Schema(type=Union[Config, bool], optional=True),
        "batch_size": Schema(int, optional=True, default=256),
        "num_workers": Schema(int, optional=True, default=os.cpu_count()),
        "epochs": Schema(int, optional=True, default=100, aliases=["epoch"]),
        "save_interval": Schema(int, optional=True, default=1),
        "trackers": Schema(type=Config, optional=True, default={}),
        "metrics": Schema(
            type=list[Config],
            optional=True,
            default=[
                {"type": "map"},
                {"type": "dice"},
                {"type": "roc_auc"},
                # {"type": "mssim"},
            ],
        ),
    }

    def __init__(self, model, train_dataset, val_dataset, force_device=None) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        if force_device is not None:
            self.device = torch.device(force_device)
            self.gpu_id = 0
        else:
            self.gpu_id = get_rank()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info(f"Device: {self.device}")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader = self._create_dataloader(
            self.train_dataset, is_train=True
        )
        self.val_dataloader = self._create_dataloader(self.val_dataset, is_train=False)

        self.model = model.to(self.device)
        self.optimizer = BaseOptimizer.from_config(
            self.optimizer.copy(), params=self.model.parameters()
        )
        if self.scheduler is not None:
            self.scheduler = BaseScheduler.from_config(
                self.scheduler.copy(), optimizer=self.optimizer
            )
        if self.early_stopper is not None and self.early_stopper is not False:
            self.early_stopper = EarlyStopping.from_config(
                self.early_stopper
                if not isinstance(self.early_stopper, bool)
                else {"Type": "EarlyStopping"}
            )

        # Initialize tracker
        self.tracker = Trackers(
            self.trackers,
            os.path.join(
                self.global_config["output_dir"], self.global_config["run_name"]
            ),
        )

        self.metrics_fn = Metrics(self.metrics)

        # Initialize training state
        self.epochs_run = 0
        self.best_loss = float("inf")

        # Wrap model with DistributedDataParallel if using multiple GPUs
        if torch.cuda.device_count() >= 2:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.gpu_id], output_device=self.gpu_id
            )
            self.logger.debug(
                f"Model wrapped with DistributedDataParallel on GPU {self.gpu_id}"
            )

        # self.loss_fn = torch.nn.BCELoss()
        self.loss_fn = BinaryCrossEntropyDiceSum()

    def preconditions(self):
        assert self.epochs > 0, "Number of epochs must be greater than 0"
        assert self.batch_size > 0, "Batch size must be greater than 0"
        assert self.num_workers > 0, "Number of workers must be greater than 0"
        self.logger.debug(f"Preconditions passed")

    def train(self):
        """
        Start the training process, including validation and test phases.
        """
        if is_main_gpu():
            self.tracker.init()
            self.logger.debug(f"Tracker initialized")

        loop = tqdm(
            range(self.epochs_run, self.epochs),
            desc="Training",
            unit="epoch...",
            disable=not is_main_gpu(),
        )
        self.model.train()
        for epoch in loop:
            train_loss = self.run_loop_train(epoch)
            if hasattr(self, "val_dataloader"):
                val_loss = self.run_loop_validation(epoch)

            if is_main_gpu():
                lr = self.optimizer.param_groups[0]["lr"]
                log = {
                    "train_loss": train_loss.item(),
                }
                log |= (
                    {"val_loss": val_loss.item()}
                    if hasattr(self, "val_dataloader")
                    else {}
                )
                log |= {"lr": lr}
                log |= (
                    {**self.metrics_fn.to_dict()}
                    if hasattr(self, "val_dataloader")
                    else {}
                )
                self.tracker.log(epoch, log)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_snapshot(
                        epoch,
                        os.path.join(
                            self.global_config.output_dir,
                            self.global_config.run_name,
                            "best.pt",
                        ),
                        train_loss,
                    )

                if epoch % self.save_interval == 0:
                    self._save_snapshot(
                        epoch,
                        os.path.join(
                            self.global_config.output_dir,
                            self.global_config.run_name,
                            f"save_{epoch}.pt",
                        ),
                        train_loss,
                    )

                # Save snapshot after every epoch
                self._save_snapshot(
                    epoch,
                    os.path.join(
                        self.global_config.output_dir,
                        self.global_config.run_name,
                        "last.pt",
                    ),
                    train_loss,
                )

                loop.set_postfix_str(
                    f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {lr:.6f}"
                    if hasattr(self, "val_dataloader")
                    else f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | LR: {lr:.6f}"
                )

                # Check for early stopping
                if self.early_stopper and self.early_stopper.step(val_loss):
                    self.logger.info(
                        f"Early stopping at epoch {epoch}. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}"
                    )
                    break

        # Test phase
        if hasattr(self, "test_dataloader"):
            test_loss = self.run_loop_validation(self.epochs, self.test_dataloader, "Test")
            if is_main_gpu():
                log = {"test_loss": test_loss.item()}
                self.tracker.log(self.epochs, log)
                self.logger.info(f"Test Loss: {test_loss:.6f}")

        if is_main_gpu():
            self.tracker.finish()
            self.logger.info(
                f"Training finished. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}, "
                f"saved at {os.path.join(self.global_config.output_dir, self.global_config.run_name, 'best.pt')}"
            )

    def run_loop_train(self, epoch):
        """
        Run the training loop for a given epoch.
        """
        self.model.train()  # Set model to training mode
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
            loss, idx = self._run_batch(batch)
            total_loss += loss.detach() * idx.sum()
            averaging_coef += idx

            if is_main_gpu():
                loop.set_postfix_str(f"Train Loss: {total_loss / averaging_coef:.6f}")

        # print(f"IDX: {idx}")
        avg_loss = total_loss / averaging_coef
        self.scheduler.step()
        return avg_loss

    def run_loop_validation(self, epoch, custom_dataloader=None, description="Validation"):
        """
        Run the validation or test loop for a given epoch.
        """
        if custom_dataloader is not None:
            val_dataloader = custom_dataloader
        else:
            val_dataloader = self.val_dataloader
        self.model.eval()
        total_loss = 0
        iters = len(self.val_dataloader)

        with torch.no_grad():
            loop = tqdm(
                enumerate(val_dataloader),
                total=iters,
                desc=f"Epoch {epoch}/{self.epochs} - f{description}",
                unit="batch",
                disable=not is_main_gpu(),
                leave=False,
            )
            for i, batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.model(**batch)
                target = batch["target"]
                idx = batch["labelled"] == 1
                loss = self.loss_fn(preds[idx], target[idx])
                total_loss += loss
                idx = torch.flatten(idx).detach().cpu().numpy()
                self.metrics_fn.update(
                    torch.flatten(preds)[idx].detach().cpu().numpy(),
                    torch.flatten(target)[idx].detach().cpu().numpy(),
                    idx,
                )
                if is_main_gpu():
                    loop.set_postfix_str(f"{description} Loss: {total_loss / (i + 1):.6f}")

        avg_loss = total_loss / len(val_dataloader)
        return avg_loss

    def _run_batch(self, batch):
        """
        Run a single training batch with masking and metrics update.
        """

        preds = self.model(**batch)
        target = batch["target"]
        labelled = batch["labelled"]
        idx = labelled == 1
        temp_loss = self.loss_fn(preds[idx], target[idx])
        self.optimizer.zero_grad()
        temp_loss.backward()
        self.optimizer.step()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"Gradient of {name} is None"

        idx = torch.flatten(idx).detach().cpu().numpy()
        # self.metrics_fn.update(torch.flatten(preds)[idx].detach().cpu().numpy(),
        #                        torch.flatten(target)[idx].detach().cpu().numpy(),
        #                        idx)

        temp_loss = temp_loss.detach().cpu()
        return temp_loss, idx.sum()

    def _save_snapshot(self, epoch, path, loss):
        """
        Save a snapshot of the training progress.
        """
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "TRAIN_INFO": {
                "EPOCHS_RUN": epoch,
                "BEST_LOSS": loss,
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "SCHEDULER_STATE": self.scheduler.state_dict(),
            },
            "GLOBAL_CONFIG": self.global_config.to_dict(),
        }
        torch.save(snapshot, path)
        self.logger.info(
            f"Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}"
        )

    def _create_dataloader(self, dataset, is_train=True):
        """
        Create a dataloader for the given dataset.
        """
        if torch.cuda.device_count() >= 2:
            sampler = DistributedSampler(
                dataset, rank=get_rank_num(), shuffle=is_train, drop_last=False
            )
        else:
            sampler = None

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
    def from_snapshot(cls, snapshot_path, model, train_dataset, val_dataset):
        """
        Load a snapshot of the training progress.
        """
        snapshot = torch.load(snapshot_path)
        global_config = GlobalConfig(config=snapshot["GLOBAL_CONFIG"])
        trainer = cls.from_config(
            global_config.to_dict(),
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        trainer.model.load_state_dict(snapshot["MODEL_STATE"])
        trainer.optimizer.load_state_dict(snapshot["TRAIN_INFO"]["OPTIMIZER_STATE"])
        trainer.scheduler.load_state_dict(snapshot["TRAIN_INFO"]["SCHEDULER_STATE"])
        trainer.epochs_run = snapshot["TRAIN_INFO"]["EPOCHS_RUN"]
        trainer.best_loss = snapshot["TRAIN_INFO"]["BEST_LOSS"]
        trainer.logger.info(
            f"Snapshot loaded from {snapshot_path} | Epochs run: {trainer.epochs_run} | Best loss: {trainer.best_loss}"
        )
        return trainer
