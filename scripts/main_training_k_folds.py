import argparse
import os
from lis2.pipeline import KfoldsTrainingPipeline


def parse_args():
    """
    Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Training pipeline configuration loader."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the training pipeline.",
    )
    return parser.parse_args()


def main(config_path: str, debug: bool):
    """
    Main function to execute the training pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.
        debug (bool): Whether to enable debug mode.
    """
    print(f"START with : ",os.environ["CUDA_VISIBLE_DEVICES"])
    training_pipeline = KfoldsTrainingPipeline.from_config(config_path, debug=debug)
    training_pipeline.run_training()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path, args.debug)
