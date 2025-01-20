from PNRIA.torch_c.pipeline import TrainingPipeline

from PNRIA.configs.config import load_yaml

if __name__ == "__main__":

    config = load_yaml(
        "/mnt/data/WORK/BigSF/Toolbox/PNRIA/configs/config_training.yaml"
    )

    training_pipeline = TrainingPipeline.from_config(config["TrainingPipeline"])
    training_pipeline.run_training()
