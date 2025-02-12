import os
from pathlib import Path
from pprint import pprint
from typing import Union, Tuple, Dict, Any, Optional

import astropy.io.fits as fits
from configurable import Configurable, Schema, Config

from src.controller import FoldsController
from src.segmenter import Segmenter
from src.trainer import Trainer


class KfoldsTrainingPipeline(Configurable):
    """
    A pipeline for training models using K-Fold cross-validation.

    This class manages the configuration and execution of training across multiple folds,
    including dataset parsing, training, and inference.

    Attributes:
        run_name (str): Name of the training run.
        inference_source (Union[Path, str], optional): Path to the inference data source.
        train_output_dir (Union[Path, str]): Directory to save training outputs.
        nb_folds (int): Number of folds for cross-validation.
        data (Config): Configuration for the dataset.
        trainer (Config): Configuration for the trainer.
        model (Config): Configuration for the model.

    Methods:
        preconditions(): Checks preconditions for the inference source.
        run_training(): Executes training across multiple folds defined by the folds controller.
        run_inference(): Runs inference using the trained model snapshot on the test dataset.
    """

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

    def __init__(self) -> None:
        """
        Initializes the KfoldsTrainingPipeline with configuration parsing and setup.
        """
        self.data = DataConfig.from_config(self.data)
        self.folds_controller_config, self.train_config, self.valid_config, self.test_config = self.data.parse_datasets_config()
        self.folds_controller = FoldsController.from_config(self.folds_controller_config)
        self.trainer["output_dir"] = self.train_output_dir

        os.makedirs(self.train_output_dir, exist_ok=True)
        self.save_dict_to_yaml(self.config, Path(self.train_output_dir) / "config_pipeline.yaml")

    def preconditions(self) -> None:
        """
        Checks preconditions for the inference source.
        """
        if self.inference_source is not None:
            self.inference_source = Path(self.inference_source) if isinstance(self.inference_source,
                                                                              str) else self.inference_source
            assert self.inference_source.exists(), f"{self.inference_source} does not exist"
            assert self.inference_source.suffix == ".fits", f"{self.inference_source} is not a FITS file"

    def run_training(self) -> None:
        """
        Executes training across multiple folds defined by the folds controller.

        This method iterates through each fold, updating dataset configurations and trainer
        settings accordingly. It trains a model for each fold and logs the results. If
        inference is enabled, predictions from each fold are aggregated and saved as a FITS file.

        Steps:
            1. Iterate through the predefined folds.
            2. Update dataset configurations for training, validation, and testing.
            3. Configure and initialize a trainer for the current fold.
            4. Train the model and log the results.
            5. If inference is enabled, generate and aggregate predictions across folds.
            6. Save the aggregated predictions to a FITS file.

        Returns:
            None
        """
        splits = self.folds_controller.splits
        fold_assignments = self.folds_controller.fold_assignments
        aggregated_predictions = None

        for fold_index, split in enumerate(splits):
            self.logger.info(f"## Running training on fold {fold_index + 1}/{len(splits)}\n")

            train_split, valid_split, test_split = split

            self.data.update_configs_for_fold(fold_assignments, train_split, valid_split, test_split)

            self.trainer.update({
                "run_name": f"{self.run_name}_fold_{fold_index}",
                "name": f"trainer_fold_{fold_index}",
                "train_dataset": self.data.trainset,
                "val_dataset": self.data.validset,
                "test_dataset": self.data.testset,
                "model": self.model,
            })

            trainer = Trainer.from_config(self.trainer)
            training_results = trainer.train()
            self.logger.info(f"Training results for fold {fold_index + 1}:")
            pprint(training_results)

            if self.inference_source is not None:
                fold_predictions = self.run_inference(training_results["final_model_path"], self.data.testset)

                if aggregated_predictions is None:
                    aggregated_predictions = fold_predictions
                else:
                    aggregated_predictions += fold_predictions

        if aggregated_predictions is not None:
            self.logger.info("Final aggregated predictions saved to aggregated_predictions.fits")
            fits.writeto(Path(self.train_output_dir) / "aggregated_predictions.fits",
                         data=aggregated_predictions,
                         header=fits.getheader(self.inference_source),
                         overwrite=True)

    def run_inference(self, model_snapshot: str, test_config: Dict[str, Any]) -> Optional[Any]:
        """
        Runs inference using the trained model snapshot on the test dataset.

        Args:
            model_snapshot (str): Path to the trained model snapshot.
            test_config (Dict[str, Any]): Configuration for the test dataset.

        Returns:
            Optional[Any]: Inference results, if available.
        """
        inference_config = {
            "model_snapshot": model_snapshot,
            "source": self.inference_source,
            "dataset": test_config,
        }

        segmenter = Segmenter.from_config(inference_config)
        results = segmenter.segment()

        return results


# region Utils

class DataConfig(Configurable):
    """
    Configuration for the datasets used in the training pipeline.

    Attributes:
        dataset_path (Union[Path, str]): Path to the dataset.
        type (str): Type of the dataset.
        controller (Dict[str, Any]): Configuration for the folds controller.
        trainset (Dict[str, Any]): Configuration for the training dataset.
        validset (Dict[str, Any]): Configuration for the validation dataset.
        testset (Dict[str, Any]): Configuration for the test dataset.

    Methods:
        preconditions(): Checks preconditions for the dataset configuration.
        parse_datasets_config(): Parses the dataset configuration for training, validation, and testing.
        update_configs_for_fold(): Updates the configurations for a specific fold.
    """

    config_schema = {
        "dataset_path": Schema(Union[Path, str]),
        "type": Schema(str),
        "controller": Schema(type=Config),
        "trainset": Schema(type=Config),
        "validset": Schema(type=Config),
        "testset": Schema(type=Config),
    }

    def preconditions(self) -> None:
        """
        Checks preconditions for the dataset configuration.
        """
        assert Path(self.dataset_path).exists(), f"Dataset path {self.dataset_path} does not exist"

    def parse_datasets_config(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Parses the dataset configuration for training, validation, and testing.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Parsed configurations for the folds controller, train, validation, and test datasets.
        """
        train_config = self.trainset
        valid_config = self.validset
        test_config = self.testset
        folds_controller_config = self.controller
        dataset_type = self.type
        dataset_path = self.dataset_path

        for config in [train_config, valid_config, test_config]:
            config["type"] = dataset_type
            config["dataset_path"] = dataset_path

        folds_controller_config["dataset_path"] = dataset_path

        return folds_controller_config, train_config, valid_config, test_config

    def update_configs_for_fold(self, fold_assignments: Dict[str, Any], train_split: Any, valid_split: Any,
                                test_split: Any) -> None:
        """
        Updates the configurations for a specific fold.

        Args:
            fold_assignments (Dict[str, Any]): Fold assignments.
            train_split (Any): Training split.
            valid_split (Any): Validation split.
            test_split (Any): Test split.
        """
        for dataset, split in zip([self.trainset, self.validset, self.testset], [train_split, valid_split, test_split]):
            dataset.update({
                "fold_assignments": fold_assignments,
                "fold_list": split,
            })

# endregion
