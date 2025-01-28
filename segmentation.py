import argparse

from torch_c.segmenter import Segmenter


def parse_args():
    """
    Parse command-line arguments for the segmentation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Segmentation pipeline configuration loader.")
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug mode for the segmentation pipeline."
    )
    return parser.parse_args()


def main(config_path: str, debug: bool):
    """
    Main function to execute the segmentation pipeline.

    Args:
        config_path (str): Path to the YAML configuration file.
        debug (bool): Whether to enable debug mode.
    """
    segmenter = Segmenter.from_config(config_path, debug=debug)
    segmenter.segment()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path, args.debug)
