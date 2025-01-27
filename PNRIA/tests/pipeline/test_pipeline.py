import unittest

from PNRIA.tests.config.config import PATH_TO_SAMPLE_DATASET, TempDir
from PNRIA.torch_c.models.custom_model import BaseModel
from PNRIA.torch_c.pipeline import TrainingPipeline
from PNRIA.utils.preprocessing import BasePatchExtraction


class TestTrainingPipeline(TempDir):

    def pipeline_config(self):
        config_dict = {
            "TrainingPipeline": {
                "run_name": "run",
                "train_output_dir": self.temp_dir,
                "data": {
                    "dataset_path": self.temp_dir / "patches.h5",
                    "type": "FilamentsDataset",
                    "controler": {
                        "train_ratio": 0.5,
                        "indices_path": self.temp_dir / "indices.pkl",
                        "save_indices": True,
                        "nb_folds": 4,  # Default is 1
                        "area_size": 64,
                        "patch_size": 32,
                    },
                    "trainset": {
                        "learning_mode": "conservative",
                        "data_augmentation": "noise",
                        "normalization_mode": "test",
                        "toEncode": ["positions"],
                        "stride": 3,
                    },
                    "validset": {
                        "learning_mode": "conservative",
                        "data_augmentation": "noise",
                        "normalization_mode": "test",
                        "toEncode": ["positions"],
                        "stride": 1,
                    },
                    "testset": {
                        "learning_mode": "conservative",
                        "data_augmentation": "noise",
                        "normalization_mode": "test",
                        "toEncode": ["positions"],
                        "stride": 1,
                    },
                },
                "trainer": {
                    "output_dir": self.temp_dir,
                    "epoch": 2,
                    "optimizer": {
                        "type": "Adam",
                        "lr": 0.005,
                    },
                    "scheduler": {
                        "type": "MultiStepLR",
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
                    "encoder": "/home/cloud-user/work/Toolbox/PNRIA/configs/encoderLin.yml",
                    "encoder_cat_position": "middle",
                },
            },
        }
        return config_dict

    def controller_config(self):
        config_dict = {
            "train_ratio": 0.5,
            "dataset_path": self.temp_dir / "patches.h5",
            "indices_path": self.temp_dir / "indices.pkl",
            "save_indices": True,
            "nb_folds": 4,
            "area_size": 64,
            "patch_size": 32,
        }

        return config_dict

    def dataset_config(self):
        config_dict = self.pipeline_config()

        # Ground truth
        controler_config_truth = config_dict["TrainingPipeline"]["data"]["controler"]
        train_config_truth = config_dict["TrainingPipeline"]["data"]["trainset"]
        valid_config_truth = config_dict["TrainingPipeline"]["data"]["validset"]
        test_config_truth = config_dict["TrainingPipeline"]["data"]["testset"]

        dataset_type = config_dict["TrainingPipeline"]["data"]["type"]
        dataset_path = config_dict["TrainingPipeline"]["data"]["dataset_path"]

        train_config_truth["type"] = dataset_type
        valid_config_truth["type"] = dataset_type
        test_config_truth["type"] = dataset_type

        controler_config_truth["dataset_path"] = dataset_path
        train_config_truth["dataset_path"] = dataset_path
        valid_config_truth["dataset_path"] = dataset_path
        test_config_truth["dataset_path"] = dataset_path

        return (
            controler_config_truth,
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

    def test_1_pipeline_init(self):
        preprocessing_config = self.preprocessing_config()
        preprocessor = BasePatchExtraction.from_config(preprocessing_config)
        preprocessor.extract_patches()

        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])
        self.assertEqual(pipeline.run_name, "run")
        self.assertEqual(pipeline.train_output_dir, self.temp_dir)

    def test_2_dataset_config_parsing(self):

        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])

        # Ground truth
        (
            controler_config_truth,
            train_config_truth,
            valid_config_truth,
            test_config_truth,
        ) = self.dataset_config()

        self.assertEqual(pipeline.folds_controler_config, controler_config_truth)
        self.assertEqual(pipeline.train_config, train_config_truth)
        self.assertEqual(pipeline.valid_config, valid_config_truth)
        self.assertEqual(pipeline.test_config, test_config_truth)

    # TODO
    # Vérifier les weights
    def test_3_run_training(self):

        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])
        pipeline.run_training()
        directory = self.temp_dir

        run_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and folder.name.startswith("run_")
        ]

        assert len(run_folders) == 4
