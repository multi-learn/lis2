import unittest
from PNRIA.torch_c.pipeline import TrainingPipeline
from PNRIA.torch_c.dataset import BaseDataset, FoldsController
from PNRIA.torch_c.models.custom_model import BaseModel
from pathlib import Path

from PNRIA.tests.config.config import PATH_TO_SAMPLE_DATASET


class TestTrainingPipeline(unittest.TestCase):

    def pipeline_config(self):
        config_dict = {
            "TrainingPipeline": {
                "run_name": "run",
                "train_output_dir": PATH_TO_SAMPLE_DATASET,
                "data": {
                    "dataset_path": PATH_TO_SAMPLE_DATASET + "patches.h5",
                    "type": "FilamentsDataset",
                    "controler": {
                        "train_ratio": 0.5,
                        "indices_path": PATH_TO_SAMPLE_DATASET + "indices.pkl",
                        "save_indices": True,
                        "nb_folds": 4,  # Default is 1
                        "area_size": 64,
                        "patch_size": 32,
                    },
                    "trainset": {
                        "learning_mode": "conservative",
                        "data_augmentation": "noise",
                        "normalization_mode": "test",
                        "input_data_noise": 0.0,
                        "output_data_noise": 0.0,
                        "toEncode": ["positions"],
                        "stride": 3,
                    },
                    "validset": {
                        "learning_mode": "conservative",
                        "data_augmentation": "noise",
                        "normalization_mode": "test",
                        "input_data_noise": 0.0,
                        "output_data_noise": 0.0,
                        "toEncode": ["positions"],
                        "stride": 1,
                    },
                    "testset": {
                        "learning_mode": "conservative",
                        "data_augmentation": "noise",
                        "normalization_mode": "test",
                        "input_data_noise": 0.0,
                        "output_data_noise": 0.0,
                        "toEncode": ["positions"],
                        "stride": 1,
                    },
                },
                "trainer": {
                    "type": "TODO",
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
            "dataset_path": PATH_TO_SAMPLE_DATASET + "patches.h5",
            "indices_path": PATH_TO_SAMPLE_DATASET + "indices.pkl",
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

    def model_ground_truth(self):
        config_dict = self.pipeline_config()
        model = BaseModel.from_config(config_dict["TrainingPipeline"]["model"])
        return model

    def test_pipeline_init(self):
        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])
        self.assertEqual(pipeline.run_name, "run")
        self.assertEqual(pipeline.train_output_dir, PATH_TO_SAMPLE_DATASET)
        self.assertEqual(pipeline.data, {})

    def test_dataset_config_parsing(self):
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

    # Peut mieux faire ?
    def test_instanciate_model(self):
        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])
        model_ground_truth = self.model_ground_truth()
        assert pipeline.model.name == model_ground_truth.name

    def test_intanciate_trainer(self):
        # Ground truth
        (
            _,
            train_config_truth,
            valid_config_truth,
            _,
        ) = self.dataset_config()

        config_dict_controller = self.controller_config()
        controller = FoldsController.from_config(config_dict_controller)

        splits = controller.generate_kfold_splits(controller.k, controller.k_train)

        area_groups, fold_assignments = controller.create_folds_random_by_area(
            k=controller.k,
            area_size=controller.area_size,
            patch_size=controller.patch_size,
            overlap=controller.overlap,
        )

        train_split, valid_split, test_split = splits[0]

        train_config_truth["fold_assignments"] = fold_assignments
        train_config_truth["fold_list"] = train_split

        valid_config_truth["fold_assignments"] = fold_assignments
        valid_config_truth["fold_list"] = valid_split

        # Ground truth
        train_dataset = BaseDataset.from_config(train_config_truth)
        val_dataset = BaseDataset.from_config(valid_config_truth)
        model = self.model_ground_truth()

        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])
        pipeline.trainer["run_name"] = "test"
        trainer = pipeline.instanciate_trainer(model, train_dataset, val_dataset)
        assert trainer.model is not None, "Model should be initialized"
        assert trainer.optimizer is not None, "Optimizer should be initialized"
        assert trainer.scheduler is not None, "Scheduler should be initialized"

    # TODO
    # Vérifier les weights
    def test_run_training(self):
        config_dict = self.pipeline_config()
        pipeline = TrainingPipeline.from_config(config_dict["TrainingPipeline"])
        pipeline.run_training()
        directory = Path(PATH_TO_SAMPLE_DATASET)

        run_folders = [
            folder
            for folder in directory.iterdir()
            if folder.is_dir() and folder.name.startswith("run_")
        ]

        assert len(run_folders) == 4
