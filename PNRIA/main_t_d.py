import torch
import matplotlib

from torch.utils.data import DataLoader

from PNRIA.configs.config import load_yaml
from PNRIA.dataset import BaseDataset



config = load_yaml("/home/cloud-user/work/Toolbox/PNRIA/configs/config_dataset_test.yaml")

# Load the model from the configuration
dataset = BaseDataset.from_config(config['dataset'])
for element in dataset.data:
    print(element)
batch_size = 1

dataloader = DataLoader(dataset, num_workers=1, batch_size=batch_size, shuffle=False)

for sample in dataloader:
    print(sample)
    break
