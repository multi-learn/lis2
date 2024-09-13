"""1D CNN implementation.

blabla
"""
from torch import nn
import torch

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layers = nn.Sequential(
            *[
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=387, stride=1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=64, out_features=1),
            ]
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        output = self.layers(x)
        return torch.squeeze(torch.sigmoid(output), dim=1)