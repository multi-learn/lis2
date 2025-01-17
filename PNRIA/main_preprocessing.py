"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""

from PNRIA.configs.config import load_yaml
from PNRIA.utils.preprocessing import BasePreprocessing

if __name__ == "__main__":
    use_case = ""
    config = load_yaml(
        "/home/cloud-user/work/Toolbox/PNRIA/configs/config_preprocessing.yml"
    )

    # Mosaic building
    if use_case == "mosaic":

        # Load the model from the configuration
        preprocessor = BasePreprocessing.from_config(config["preprocessing_mosaics"])
        preprocessor.create_folds()

    # Extract all patches in one file
    else:

        preprocessor = BasePreprocessing.from_config(config["preprocessing_patches"])
        preprocessor.extract_patches()
