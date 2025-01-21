from pathlib import Path
from typing import Union

from PNRIA.configs.config import (
    Schema,
    Customizable,
    Config,
)
from PNRIA.torch_c.controleur import FoldsController
from PNRIA.torch_c.trainer import Trainer


class TrainingPipeline(Customizable):

    config_schema = {
        "run_name": Schema(str),
        "train_output_dir": Schema(Union[Path, str]),
        "nb_folds": Schema(int, default=1),
        "data": Schema(type=Config),
        "trainer": Schema(type=Config),
        "model": Schema(type=Config),
    }

    def __init__(self):
        (
            self.folds_controler_config,
            self.train_config,
            self.valid_config,
            self.test_config,
        ) = self.parse_datasets_config()
        self.folds_controler = FoldsController.from_config(self.folds_controler_config)
        self.trainer["output_dir"] = self.train_output_dir

    def parse_datasets_config(self):
        train_config = self.data.get("trainset")
        valid_config = self.data.get("validset")
        test_config = self.data.get("testset")
        folds_controler_config = self.data.get("controler")

        dataset_type = self.data.get("type")
        dataset_path = self.data.get("dataset_path")

        train_config["type"] = dataset_type
        valid_config["type"] = dataset_type
        test_config["type"] = dataset_type

        folds_controler_config["dataset_path"] = dataset_path

        train_config["dataset_path"] = dataset_path
        valid_config["dataset_path"] = dataset_path
        test_config["dataset_path"] = dataset_path

        return folds_controler_config, train_config, valid_config, test_config

    def run_training(self):

        splits = self.folds_controler.splits
        fold_assignments = self.folds_controler.fold_assignments
        for idx, split in enumerate(splits):

            self.logger.info(f"Running training on split number {idx} on {len(splits)}")
            train_split, valid_split, test_split = split

            self.train_config["fold_assignments"] = fold_assignments
            self.train_config["fold_list"] = train_split

            self.valid_config["fold_assignments"] = fold_assignments
            self.valid_config["fold_list"] = valid_split

            self.test_config["fold_assignments"] = fold_assignments
            self.test_config["fold_list"] = test_split

            self.trainer["run_name"] = self.run_name + f"_fold_{idx}"
            self.trainer["name"] = f"trainer_fold_{idx}"
            self.trainer["train_dataset"] = self.train_config
            self.trainer["val_dataset"] = self.valid_config
            self.trainer["test_dataset"] = self.test_config

            # Besoin que ca soit le meme model a chaque fois pour les splits??
            self.trainer["model"] = self.model

            trainer = Trainer.from_config(
                self.trainer,
            )
            print(trainer.train())
