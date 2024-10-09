import os
import logging
from typing import Union

import torch
import matplotlib
from torch.utils.data import random_split
from PNRIA.configs.config import Customizable, Schema, Config, GlobalConfig
from PNRIA.dataset import BaseDataset
from PNRIA.torch_c.early_stop import EarlyStopping
from PNRIA.torch_c.metrics import Metrics
from PNRIA.torch_c.models.custom_model import BaseModel
from PNRIA.torch_c.optim import BaseOptimizer
from PNRIA.torch_c.scheduler import BaseScheduler
from PNRIA.torch_c.trackers import Trackers
from PNRIA.utils.distributed import get_rank, get_rank_num, is_main_gpu, synchronize

matplotlib.use('TkAgg')
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch import distributed as dist, nn


class Trainer(Customizable):
    config_schema = {
        'model': Schema(type=Config),
        'dataset': Schema(type=Config),
        'optimizer': Schema(type=Config),
        'scheduler': Schema(type=Config, optional=True),
        'early_stopping': Schema(type=Union[Config, bool], optional=True),
        'split_ratio': Schema(float, optional=True, default=0.8),
        'batch_size': Schema(int, optional=True, default=8),
        'num_workers': Schema(int, optional=True, default=os.cpu_count()),
        'epochs': Schema(int, optional=True, default=100),
        'save_interval': Schema(int, optional=True, default=10),
        'trackers': Schema(type=Config, optional=True, default={}),
        'metrics': Schema(type=list[Config], optional=True, default=[{'type': 'accuracy'},
                                                                     {'type': 'dice'},
                                                                     {'type': 'seg_acc'}]),
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

        # Create dataloaders
        self.train_dataloader = self._create_dataloader(self.train_dataset, is_train=True)
        self.val_dataloader = self._create_dataloader(self.val_dataset, is_train=False)

        # Initialize model, optimizer, scheduler, and early stopping
        self.model = BaseModel.from_config(self.model, image_size=self.dataset.image_size).to(self.device)
        self.optimizer = BaseOptimizer.from_config(self.optimizer.copy(), self.model.parameters())
        if self.scheduler is not None:
            self.scheduler = BaseScheduler.from_config(self.scheduler.copy(), self.optimizer,
                                                       steps_per_epoch=len(self.train_dataloader))
        if self.early_stopping is not None:
            self.early_stopper = EarlyStopping.from_config(
                self.early_stopping if isinstance(bool, self.early_stopping) else {})

        # Initialize tracker
        self.tracker = Trackers(self.tracking, self.global_config["output_dir"])

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

    def train(self):
        """
        Start the training process.
        """
        if is_main_gpu():
            self.tracker.init()
            loop = tqdm(
                range(self.epochs_run, self.epochs),
                desc="Training",
                unit="epoch",
                dynamic_ncols=True,
            )
        else:
            loop = range(self.epochs_run, self.epochs)

        for epoch in loop:
            train_loss = self._run_epoch(epoch, self.train_dataloader)
            # val_loss = self._run_epoch(epoch, self.val_dataloader, training=False)

            if is_main_gpu():
                lr = self.optimizer.param_groups[0]['lr']
                log = {
                    "train_loss": train_loss.item(),
                    "lr": lr,
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
                self._save_snapshot(
                    epoch,
                    os.path.join(self.global_config.output_dir, self.global_config.run_name, "last.pt"),
                    train_loss,
                )
                loop.set_postfix_str(
                    f"Epoch: {epoch} | Train Loss: {train_loss:.5f} | Val Loss: {train_loss:.5f} | LR: {lr:.6f}"
                )
                if self.early_stopper.step(train_loss):
                    self.logger.info(
                        f"Early stopping at epoch {epoch}. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}"
                    )
                    break

        if is_main_gpu():
            self.tracker.finish()
            self.logger.info(
                f"Training finished. Best loss: {self.best_loss:.6f}, LR: {lr:.6f}, "
                f"saved at {os.path.join(self.global_config.output_dir, self.global_config.run_name, 'best.pt')}"
            )

    def _run_epoch(self, epoch, dataloader):
        """
        Run a training or validation epoch.
        """
        iters = len(dataloader)
        if dist.is_initialized():
            dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        self.metrics_fn.reset()  # Reset metrics at the beginning of each epoch
        loop = tqdm(
            enumerate(dataloader),
            total=iters,
            desc=f"Epoch {epoch}/{self.epochs + self.epochs_run} - {'Training'}",
            unit="batch",
            leave=False,
            postfix="",
            disable=not is_main_gpu(),
        )
        for i, batch in loop:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss = self._run_batch(batch)
            total_loss += loss

            if is_main_gpu():
                loop.set_postfix_str(f"Loss : {total_loss / (i + 1):.6f}")
                log = {
                    "avg_loss_it": loss.item(),
                    "lr_it": self.optimizer.param_groups[0]["lr"],
                }
                log.update({f"avg_{k}_it": v for k, v in self.metrics_fn.compute().items()})
                self.tracker.log(i, log)

            if self.scheduler.step_each_batch:
                self.scheduler.step()

        self.logger.debug(
            f"Epoch {epoch} | Batchsize: {self.batch_size} | Steps: {len(dataloader) * epoch} | "
            f"Last loss: {total_loss / len(dataloader)} | "
            f"Lr : {self.optimizer.param_groups[0]['lr']} | ",
            f"Metrics: {self.metrics_fn.compute()}"
        )
        self.scheduler.step(total_loss / len(dataloader))

        # if epoch % self.save_interval == 0 and is_main_gpu():
        #     samples = self.model.sample(batch_size=4, condition=batch["condition"][:4])
        #     self.plot_grid(f"samples_grid_{epoch}.jpg", samples.cpu().numpy())

        if epoch % self.save_interval == 0:
            synchronize()

        return total_loss / len(dataloader)

    def _run_batch(self, batch):
        """
        Run a single training batch.
        """
        self.optimizer.zero_grad()
        preds = self.model(**batch)
        loss = self.loss_fn(preds, batch['targets'])
        self.metrics_fn.update(preds, **batch)
        loss.backward()
        self.optimizer.step()
        loss = loss.detach().cpu()
        return loss

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
                "OPTIMIZER_CONFIG": self.optimizer.to_config(),
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "SCHEDULER_CONFIG": self.scheduler.to_config(),
                "SCHEDULER_STATE": self.scheduler.state_dict(),
            },
            "DATAPROCESS": self.dataset.to_config(),
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
