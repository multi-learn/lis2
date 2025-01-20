from pathlib import Path
from typing import Union

from PNRIA.torch_c.controleur import FoldsController
from PNRIA.torch_c.models.custom_model import BaseModel
from PNRIA.torch_c.trainer import Trainer
from PNRIA.configs.config import (
    Schema,
    Customizable,
    Config,
)
from PNRIA.torch_c.dataset import BaseDataset


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
        self.model = BaseModel.from_config(self.model)
        self.trainer["output_dir"] = self.train_output_dir

    def instanciate_trainer(self, model, train_dataset, val_dataset):
        GlobalConfig(self.trainer)
        trainer = Trainer.from_config(
            self.trainer,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        return trainer

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

        splits = FoldsController.generate_kfold_splits(
            self.folds_controler.k, self.folds_controler.k_train
        )

        area_groups, fold_assignments = (
            self.folds_controler.create_folds_random_by_area()
        )

        for idx, split in enumerate(splits):

            self.logger.info(f"Running training on split number {idx} on {len(splits)}")
            train_split, valid_split, test_split = split

            config_train_loop = self.train_config
            config_valid_loop = self.valid_config
            config_test_loop = self.test_config

            config_train_loop["fold_assignments"] = fold_assignments
            config_train_loop["fold_list"] = train_split

            config_valid_loop["fold_assignments"] = fold_assignments
            config_valid_loop["fold_list"] = valid_split

            config_test_loop["fold_assignments"] = fold_assignments
            config_test_loop["fold_list"] = test_split

            train_dataset = BaseDataset.from_config(config_train_loop)
            val_dataset = BaseDataset.from_config(config_valid_loop)
            test_dataset = BaseDataset.from_config(config_test_loop)

            # Start training here
            self.trainer["run_name"] = self.run_name + f"_fold_{idx}"
            trainer = self.instanciate_trainer(self.model, train_dataset, val_dataset)
            trainer.train()
