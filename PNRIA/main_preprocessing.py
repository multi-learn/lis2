"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""

from PNRIA.configs.config import load_yaml
from PNRIA.utils.preprocessing import BasePatchExtraction

if __name__ == "__main__":
    config = load_yaml(
        "/mnt/data/WORK/BigSF/Toolbox/PNRIA/configs/config_preprocessing.yml"
    )

    preprocessor = BasePatchExtraction.from_config(config["preprocessing_patches"])
    preprocessor.extract_patches()
