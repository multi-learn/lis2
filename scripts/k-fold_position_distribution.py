import os
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from deep_filaments.io.dataset import FilamentsDataset
from torch.utils.data import DataLoader

from deep_filaments.io.utils import get_sorted_file_list

path = "/home/loris/PhD/Dev/Datasets/nh2_dataset/k_folds/train_sets"
train_dataset_files = get_sorted_file_list(path)
fig = go.Figure()
for i in range(len(train_dataset_files)):
    dataset = FilamentsDataset(os.path.join(path, train_dataset_files[i]))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    longitude = []
    for j, samples in enumerate(dataloader):
        position = samples["position"]
        for k in range(position.shape[0]):
            longitude.append(torch.squeeze(position[k, 1, 0]).numpy())
    longitude = ((57000 - np.array(longitude)) * 0.00319444444400 + 180)
    fig.add_trace(go.Histogram(x=np.array(longitude), name=f"Fold {i}", xbins=dict(start=0, end=350, size=10)))
fig.update_layout(xaxis_title="Longitude in degree", 
                  yaxis_title="Number of patches", 
                  bargap=0.2,
                  bargroupgap=0.1,
                  title="Longitude histogram of the training set of the 10 folds used for Galactic plane segmentation")
fig.show()

path = "/home/loris/PhD/Dev/Datasets/nh2_dataset/k_folds/train_sets/fold_0_train.h5"
dataset = FilamentsDataset(path)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
longitude = []
fig = go.Figure()
for j, samples in enumerate(dataloader):
    position = samples["position"]
    for k in range(position.shape[0]):
        longitude.append(torch.squeeze(position[k, 1, 0]).numpy())
longitude = ((57000 - np.array(longitude)) * 0.00319444444400 + 180)
fig.add_trace(go.Histogram(x=longitude, xbins=dict(start=0, end=350, size=10)))
fig.update_layout(xaxis_title="Longitude in degree", 
                  yaxis_title="Number of patches", 
                  bargap=0.2, 
                  title="Longitude histogram of the training set of fold 0")
fig.show()