import torch
import matplotlib

from torch.utils.data import DataLoader

from PNRIA.configs.config import load_yaml
from PNRIA.dataset import BaseDataset



config = load_yaml("/mnt/data/WORK/BigSF/Toolbox/PNRIA/configs/config_dataset_test.yaml")

# Load the model from the configuration
dataset = BaseDataset.from_config(config['dataset'])
batch_size = 1

dataloader = DataLoader(dataset, num_workers=1, batch_size=batch_size, shuffle=False)

for sample in dataloader:
    print(sample)
    break