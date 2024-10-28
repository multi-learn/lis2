import os
import logging
from typing import Union

import torch
import matplotlib
from torch.utils.data import random_split
from PNRIA.configs.config import Customizable, Schema, Config, GlobalConfig
from PNRIA.dataset import BaseDataset
from PNRIA.torch_c.early_stop import EarlyStopping
from PNRIA.torch_c.loss import BinaryCrossEntropyDiceSum
from PNRIA.torch_c.metrics import Metrics
from PNRIA.torch_c.models.custom_model import BaseModel
from PNRIA.torch_c.optim import BaseOptimizer
from PNRIA.torch_c.scheduler import BaseScheduler
from PNRIA.torch_c.trackers import Trackers
from PNRIA.utils.distributed import get_rank, get_rank_num, is_main_gpu, synchronize

matplotlib.use('TkAgg')
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from torch import distributed as dist, nn

from abc import ABC, abstractmethod


class ITrainer(ABC, Customizable):

    @abstractmethod
    def train(self) -> None:
        pass


class Trainer(ITrainer):
    config_schema = {
        'output_dir': Schema(str),
        'run_name': Schema(str),
        'model': Schema(type=Config),
        'dataset': Schema(type=Config),
        'dataset_test': Schema(type=Config, optional=True),
        'optimizer': Schema(type=Config),
        'scheduler': Schema(type=Config, optional=True),
        'early_stopper': Schema(type=Union[Config, bool], optional=True),
        'split_ratio': Schema(float, optional=True, default=0.8),
        'batch_size': Schema(int, optional=True, default=64),
        'num_workers': Schema(int, optional=True, default=os.cpu_count()),
        'epochs': Schema(int, optional=True, default=100, aliases=['epoch']),
        'save_interval': Schema(int, optional=True, default=10),
        'trackers': Schema(type=Config, optional=True, default={}),
        'metrics': Schema(type=list[Config], optional=True, default=[{'type': 'map'},
                                                                     {'type': 'dice'},
                                                                     {'type': 'roc_auc'},
                                                                     # {'type': 'mssim'},
        ]),
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.gpu_id = get_rank()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Split the dataset into train and validation datasets
        self.dataset = BaseDataset.from_config(self.dataset)
        train_size = int(self.split_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.train_dataloader = self._create_dataloader(self.train_dataset, is_train=True)
        self.val_dataloader = self._create_dataloader(self.val_dataset, is_train=False)

        self.model = BaseModel.from_config(self.model).to(self.device)
        self.optimizer = BaseOptimizer.from_config(self.optimizer.copy(), self.model.parameters())
        if self.scheduler is not None:
            self.scheduler = BaseScheduler.from_config(self.scheduler.copy(), self.optimizer)
        if self.early_stopper is not None:
            self.early_stopper = EarlyStopping.from_config(
                self.early_stopper if isinstance(bool, self.early_stopper) else {})

        # Initialize tracker
        self.tracker = Trackers(self.trackers, os.path.join(self.global_config["output_dir"], self.global_config["run_name"]))

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

        self.loss_fn = torch.nn.BCELoss()

    def train(self):
        """
        Start the training process, including validation and test phases.
        """
        if is_main_gpu():
            self.tracker.init()

        # Test dataset handling
        if hasattr(self, 'dataset_test') and self.dataset_test:
            self.test_dataset = BaseDataset.from_config(self.dataset_test)
            self.test_dataloader = self._create_dataloader(self.test_dataset, is_train=False)

        loop = tqdm(
            range(self.epochs_run, self.epochs),
            desc="Training",
            unit="epoch...",
            disable=not is_main_gpu(),
        )

        for epoch in loop:
            # Run training loop
            train_loss = self.run_loop_train(epoch)

            # Optionally run validation loop
            # val_loss = self.run_loop_validation(epoch, self.val_dataloader)

            if is_main_gpu():
                lr = self.optimizer.param_groups[0]['lr']
                log = {
                    "train_loss": train_loss.item(),
                    # "val_loss": val_loss.item(),
                    "lr": lr,
                    **self.metrics_fn.to_dict(),
                }
                self.tracker.log(epoch, log)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_snapshot(
                        epoch,
                        os.path.join(self.global_config.output_dir, self.global_config.run_name, "best.pt"),
                        train_loss,
                    )

                if epoch % self.save_interval == 0:
                    self._save_snapshot(
                        epoch,
                        os.path.join(self.global_config.output_dir, self.global_config.run_name, f"save_{epoch}.pt"),
                        train_loss,
                    )

                # Save snapshot after every epoch
                self._save_snapshot(
                    epoch,
                    os.path.join(self.global_config.output_dir, self.global_config.run_name, "last.pt"),
                    train_loss,
                )

                loop.set_postfix_str(
                    # f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {lr:.6f}"
                    f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | LR: {lr:.6f}"
                )

                # Check for early stopping
                if self.early_stopper and self.early_stopper.step(train_loss):
                    self.logger.info(
                        f"Early stopping at epoch {epoch}. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}"
                    )
                    break

        # Test phase
        if hasattr(self, 'test_dataloader'):
            test_loss = self.run_loop_validation(self.epochs, self.test_dataloader)
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
            loss, idx = self._run_batch(batch)  # Run training batch and compute loss
            total_loss += loss * idx
            averaging_coef += idx

            if is_main_gpu():
                loop.set_postfix_str(f"Train Loss: {total_loss / (i + 1):.6f}")


        avg_loss = total_loss / averaging_coef
        self.scheduler.step()  # Adjust learning rate at the end of the epoch if needed
        return avg_loss

    def run_loop_validation(self, epoch, dataloader):
        """
        Run the validation or test loop for a given epoch.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        iters = len(dataloader)

        with torch.no_grad():  # Disable gradient computation for validation/test
            loop = tqdm(
                enumerate(dataloader),
                total=iters,
                desc=f"Epoch {epoch}/{self.epochs} - Validation/Test",
                unit="batch",
                disable=not is_main_gpu(),
                leave=False,
            )

            for i, batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.model(**batch)
                target = batch['target']
                loss = self.loss_fn(preds, target)
                total_loss += loss
                # self.metrics_fn.update(torch.flatten(preds).detach().cpu().numpy(),
                #                        torch.flatten(target).detach().cpu().numpy(),
                #                        torch.flatten(batch['labelled'] == 1).detach().cpu().numpy())

                if is_main_gpu():
                    loop.set_postfix_str(f"Validation/Test Loss: {total_loss / (i + 1):.6f}")

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def _run_batch(self, batch):
        """
        Run a single training batch with masking and metrics update.
        """
        self.optimizer.zero_grad()

        preds = self.model(**batch)
        target = batch['target']
        labelled = batch['labelled']
        idx = labelled == 1
        temp_loss = self.loss_fn(preds[idx], target[idx])
        temp_loss.backward()
        self.optimizer.step()
        idx = torch.flatten(idx).detach().cpu().numpy()
        self.metrics_fn.update(torch.flatten(preds)[idx].detach().cpu().numpy(),
                               torch.flatten(target)[idx].detach().cpu().numpy(),
                               idx)

        temp_loss = temp_loss.detach().cpu()
        return temp_loss, idx.sum()

    def _save_snapshot(self, epoch, path, loss):
        """
        Save a snapshot of the training progress.
        """
        snapshot = {
            "MODEL": {
                "MODEL_STATE": self.model.state_dict(),
                "MODEL_CONFIG": self.model.to_config(),
            },
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
            sampler = DistributedSampler(dataset, rank=get_rank_num(), shuffle=is_train, drop_last=False)
        else:
            sampler = None

        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                pin_memory=True,
                                persistent_workers=True,
                                shuffle=(not is_train) if sampler is None else False,
                                num_workers=self.num_workers,
                                sampler=sampler)

        return dataloader

    @classmethod
    def from_snapshot(cls, snapshot_path):
        """
        Load a snapshot of the training progress.
        """
        snapshot = torch.load(snapshot_path)
        global_config = GlobalConfig(config=snapshot["GLOBAL_CONFIG"])
        trainer = cls.from_config(global_config.to_dict()["config"])
        trainer.model.load_state_dict(snapshot["MODEL"]["MODEL_STATE"])
        trainer.optimizer.load_state_dict(snapshot["TRAIN_INFO"]["OPTIMIZER_STATE"])
        trainer.scheduler.load_state_dict(snapshot["TRAIN_INFO"]["SCHEDULER_STATE"])
        trainer.epochs_run = snapshot["TRAIN_INFO"]["EPOCHS_RUN"]
        trainer.best_loss = snapshot["TRAIN_INFO"]["BEST_LOSS"]
        trainer.logger.info(
            f"Snapshot loaded from {snapshot_path} | Epochs run: {trainer.epochs_run} | Best loss: {trainer.best_loss}"
        )
        return trainer
