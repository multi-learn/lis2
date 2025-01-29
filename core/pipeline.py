import os
from pathlib import Path
from pprint import pprint
from typing import Union

import astropy.io.fits as fits

from configs.config import (
    Schema,
    Customizable,
    Config,
)
from core.controller import FoldsController
from core.segmenter import Segmenter
from core.trainer import Trainer


class KfoldsTrainingPipeline(Customizable):
    aliases = ["kfold_pipeline"]

    config_schema = {
        "run_name": Schema(str),
        "inference_source": Schema(Union[Path, str], optional=True),
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

        os.makedirs(os.path.join(self.train_output_dir), exist_ok=True)
        self.save_dict_to_yaml(self.config, os.path.join(self.train_output_dir, "config_pipeline.yaml"))

    def preconditions(self):
        if self.inference_source is not None:
            self.inference_source = Path(self.inference_source) if isinstance(self.inference_source,
                                                                              str) else self.inference_source
            assert self.inference_source.exists(), f"{self.inference_source} does not exist"
            assert self.inference_source.suffix == ".fits", f"{self.inference_source} is not a fit file"

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
        """
        Execute training across multiple folds defined by the folds controller.
        """
        splits = self.folds_controler.splits
        fold_assignments = self.folds_controler.fold_assignments
        aggregated_predictions = None
        for fold_index, split in enumerate(splits):
            self.logger.info(f"## Running training on fold {fold_index + 1}/{len(splits)}\n")

            train_split, valid_split, test_split = split

            self.train_config.update({
                "fold_assignments": fold_assignments,
                "fold_list": train_split,
            })

            self.valid_config.update({
                "fold_assignments": fold_assignments,
                "fold_list": valid_split,
            })

            self.test_config.update({
                "fold_assignments": fold_assignments,
                "fold_list": test_split,
            })

            self.trainer.update({
                "run_name": f"{self.run_name}_fold_{fold_index}",
                "name": f"trainer_fold_{fold_index}",
                "train_dataset": self.train_config,
                "val_dataset": self.valid_config,
                "test_dataset": self.test_config,
                "model": self.model,
            })

            trainer = Trainer.from_config(self.trainer)
            training_results = trainer.train()
            self.logger.info(f"Training results for fold {fold_index + 1}:")
            pprint(training_results)
            if self.inference_source is not None:
                fold_predictions = self.run_inference(training_results["final_model_path"], self.test_config)

                if aggregated_predictions is None:
                    aggregated_predictions = fold_predictions
                else:
                    aggregated_predictions += fold_predictions

        self.logger.info("Final aggregated predictions saved to aggregated_predictions.fits")
        if aggregated_predictions is not None:
            fits.writeto(os.path.join(self.train_output_dir, "aggregated_predictions.fits"),
                         data=aggregated_predictions,
                         header=fits.getheader(self.inference_source),
                         overwrite=True,
            )

    def run_inference(self, model_snapshot: str, test_config):
        """
        Run inference using the trained model snapshot on the test dataset.

        """
        inference_config = {
            "model_snapshot": model_snapshot,
            "source": self.inference_source,
            "dataset": test_config,
        }

        segmenter = Segmenter.from_config(inference_config)
        results = segmenter.segment()

        return results
