"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""

from PNRIA.configs.config import load_yaml
from PNRIA.utils.preprocessing import BasePreprocessing

if __name__ == "__main__":
    use_case = "mosaic"
    config = load_yaml("/home/cloud-user/work/Toolbox/PNRIA/configs/config_preprocessing.yml")
    
    if use_case == "patches":

        # Load the model from the configuration
        preprocessor = BasePreprocessing.from_config(config['preprocessing'])
        preprocessor.create_folds()
        
    elif use_case == "mosaic":
        
        # Load the model from the configuration
        preprocessor = BasePreprocessing.from_config(config['preprocessing'])
        preprocessor.create_folds()
