"""
Creation of dataset from given n(h2), RoI and mask (missing data)
"""

from PNRIA.configs.config import load_yaml
from PNRIA.utils.preprocessing import BasePreprocessing

if __name__ == "__main__":
<<<<<<< HEAD
    use_case = "patches"
=======
    use_case = "mosaic"
>>>>>>> 2994ce1 (fixing dataset according to ML, synchro with modular-integration)
    config = load_yaml("/home/cloud-user/work/Toolbox/PNRIA/configs/config_preprocessing.yml")
    
    if use_case == "patches":

        # Load the model from the configuration
<<<<<<< HEAD
        preprocessor = BasePreprocessing.from_config(config['preprocessing_patches'])
=======
        preprocessor = BasePreprocessing.from_config(config['preprocessing'])
>>>>>>> 2994ce1 (fixing dataset according to ML, synchro with modular-integration)
        preprocessor.create_folds()
        
    elif use_case == "mosaic":
        
        # Load the model from the configuration
<<<<<<< HEAD
        preprocessor = BasePreprocessing.from_config(config['preprocessing_mosaics'])
=======
        preprocessor = BasePreprocessing.from_config(config['preprocessing'])
>>>>>>> 2994ce1 (fixing dataset according to ML, synchro with modular-integration)
        preprocessor.create_folds()
