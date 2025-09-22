from lis2.models.base_model import BaseModel
from lis2.pipeline import KfoldsTrainingPipeline
from lis2.preprocessing import BasePatchExtraction
from tests.config.config import PATH_TO_SAMPLE_DATASET, TempDir, PATH_PROJECT
import torch
import pytest

from unittest.mock import patch
from tests.trainer.mocks import (
    MockDataset,
    MockModel,
    MockOptimizer,
    MockScheduler,
    MockEarlyStopping,
    MockMetrics,
)


class TestTrainingPipeline(TempDir):

    def pipeline_config(self):
        config_dict = {
            "TrainingPipeline": {
                "run_name": "run",
                "train_output_dir": self.temp_dir,
                "data": {
                    "dataset_path": self.temp_dir / "patches.h5",
                    "type": "FilamentsDataset",
                    "controller": {
                        "type": "RandomController",
                        "train_ratio": 0.5,
                        "indices_path": self.temp_dir,
                        "save_indices": True,
                        "nb_folds": 4,  # Default is 1
                        "area_size": 64,
                        "patch_size": 32,
                    },
                    "trainset": {
                        "name": "MockTrain",
                        "data_augmentations": [
                            {"type": "ToTensor"},
                            {
                                "type": "NoiseDataAugmentation",
                                "name": "in_data",
                                "keys_to_augment": ["patch"],
                            },
                            {
                                "type": "NoiseDataAugmentation",
                                "name": "out_data",
                                "keys_to_augment": ["spines"],
                            },
                        ],
                        "toEncode": ["positions"],
                        "stride": 3,
                    },
                    "validset": {
                        "name": "MockValid",
                        "data_augmentations": [
                            {"type": "ToTensor"},
                            {
                                "type": "NoiseDataAugmentation",
                                "name": "in_data",
                                "keys_to_augment": ["patch"],
                            },
                            {
                                "type": "NoiseDataAugmentation",
                                "name": "out_data",
                                "keys_to_augment": ["spines"],
                            },
                        ],
                        "toEncode": ["positions"],
                        "stride": 1,
                    },
                    "testset": {
                        "name": "MockTest",
                        "data_augmentations": [
                            {"type": "ToTensor"},
                            {
                                "type": "NoiseDataAugmentation",
                                "name": "in_data",
                                "keys_to_augment": ["patch"],
                            },
                            {
                                "type": "NoiseDataAugmentation",
                                "name": "out_data",
                                "keys_to_augment": ["spines"],
                            },
                        ],
                        "toEncode": ["positions"],
                        "stride": 1,
                    },
                },
                "trainer": {
                    "output_dir": self.temp_dir,
                    "epoch": 2,
                    "optimizer": {
                        "type": "MockOptimizer",
                        "lr": 0.005,
                    },
                    "scheduler": {
                        "type": "MockScheduler",
                        "milestones": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                        "gamma": 0.1,
                    },
                },
                "model": {
                    "type": "unet",
                    "name": "unet2D_pe_alt",
                    "in_channels": 1,
                    "out_channels": 1,
                    "features": 64,
                    "dimension": 2,
                    "num_blocks": 5,
                    "encoder": str(PATH_PROJECT) + "/configs/encoder/encoderLin.yml",
                    "encoder_cat_position": "middle",
                },
            },
        }
        return config_dict

    def controller_config(self):
        config_dict = {
            "train_ratio": 0.5,
            "dataset_path": self.temp_dir / "patches.h5",
            "indices_path": self.temp_dir,
            "save_indices": True,
            "nb_folds": 4,
            "area_size": 64,
            "patch_size": 32,
        }

        return config_dict

    def dataset_config(self):
        config_dict = self.pipeline_config()

        # Ground truth
        controller_config_truth = config_dict["TrainingPipeline"]["data"]["controller"]
        train_config_truth = config_dict["TrainingPipeline"]["data"]["trainset"]
        valid_config_truth = config_dict["TrainingPipeline"]["data"]["validset"]
        test_config_truth = config_dict["TrainingPipeline"]["data"]["testset"]

        dataset_type = config_dict["TrainingPipeline"]["data"]["type"]
        dataset_path = config_dict["TrainingPipeline"]["data"]["dataset_path"]

        train_config_truth["type"] = dataset_type
        valid_config_truth["type"] = dataset_type
        test_config_truth["type"] = dataset_type

        controller_config_truth["dataset_path"] = dataset_path
        train_config_truth["dataset_path"] = dataset_path
        valid_config_truth["dataset_path"] = dataset_path
        test_config_truth["dataset_path"] = dataset_path

        return (
            controller_config_truth,
            train_config_truth,
            valid_config_truth,
            test_config_truth,
        )

    def preprocessing_config(self):
        config = {
            "type": "PatchExtraction",
            "image": PATH_TO_SAMPLE_DATASET / "sample_image.fits",
            "target": PATH_TO_SAMPLE_DATASET / "sample_target.fits",
            "missing": PATH_TO_SAMPLE_DATASET / "sample_missing.fits",
            "background": PATH_TO_SAMPLE_DATASET / "sample_background.fits",
            "output": self.temp_dir,
            "patch_size": 32,
        }
        return config

    def model_ground_truth(self):
        config_dict = self.pipeline_config()
        model = BaseModel.from_config(config_dict["TrainingPipeline"]["model"])
        return model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_1_pipeline_init(self):
        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        preprocessor.extract_patches()

        config_dict = self.pipeline_config()
        pipeline = KfoldsTrainingPipeline.from_config(config_dict["TrainingPipeline"])
        self.assertEqual(pipeline.run_name, "run")
        self.assertEqual(pipeline.train_output_dir, self.temp_dir)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_2_dataset_config_parsing(self):

        config_dict = self.pipeline_config()
        pipeline = KfoldsTrainingPipeline.from_config(config_dict["TrainingPipeline"])

        # Ground truth
        (
            controller_config_truth,
            train_config_truth,
            valid_config_truth,
            test_config_truth,
        ) = self.dataset_config()

        self.assertEqual(pipeline.folds_controller_config, controller_config_truth)
        self.assertEqual(pipeline.train_config, train_config_truth)
        self.assertEqual(pipeline.valid_config, valid_config_truth)
        self.assertEqual(pipeline.test_config, test_config_truth)

    # TODO
    # Vérifier les weights
    @patch("lis2.datasets.BaseDataset", MockDataset)
    @patch("lis2.models.base_model.BaseModel", MockModel)
    @patch("lis2.optimizer.BaseOptimizer", MockOptimizer)
    @patch("lis2.scheduler.BaseScheduler", MockScheduler)
    @patch("lis2.early_stop.BaseEarlyStopping", MockEarlyStopping)
    @patch("lis2.metrics.MetricManager", MockMetrics)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_3_run_training(self):

        config_dict = self.pipeline_config()
        pipeline = KfoldsTrainingPipeline.from_config(config_dict["TrainingPipeline"])
        pipeline.run_training()
        directory = self.temp_dir

        run_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and folder.name.startswith("run_")
        ]

        assert len(run_folders) == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_4_k1(self):
        config_dict = self.pipeline_config()
        config_dict["TrainingPipeline"]["data"]["controller"]["nb_folds"] = 1
        config_dict["TrainingPipeline"]["data"]["controller"]["k_train"] = 0.40
        config_dict["TrainingPipeline"]["run_name"] = "run_k1"

        pipeline = KfoldsTrainingPipeline.from_config(config_dict["TrainingPipeline"])
        pipeline.run_training()
        directory = self.temp_dir

        run_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and folder.name.startswith("run_k1")
        ]

        assert len(run_folders) == 1
