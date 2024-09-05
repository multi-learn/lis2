"""1D CNN implementation.

blabla
"""
from torch import nn
import torch

class VGG1D(nn.Module):
    def __init__(self):
        super(VGG1D, self).__init__()
        self.layers = nn.Sequential(
            *[
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1),
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = 2, stride = 2),
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = 2, stride = 2),
                nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = 2, stride = 2),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size = 2, stride = 2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2*7*512, out_features=4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=4096, out_features=4096),
                nn.ReLU(),
                nn.Linear(in_features=4096, out_features=1),
            ]
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        output = self.layers(x)
        return torch.squeeze(torch.sigmoid(output), dim=1)