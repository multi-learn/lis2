import argparse

from lis2.preprocessing import FilamentMosaicBuilding


def parse_args():
    """
    Parse command-line arguments for the preprocessing script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Dataset preprocessing script.")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for preprocessing."
    )
    return parser.parse_args()


def main(config_path: str, debug: bool):
    """
    Main function to load configuration and execute the patch extraction.

    Args:
        config_path (str): Path to the YAML configuration file.
        debug (bool): Whether to enable debug mode.
    """
    mosaic_builder = FilamentMosaicBuilding.from_config(config_path, debug=debug)
    mosaic_builder.mosaic_building()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path, args.debug)
