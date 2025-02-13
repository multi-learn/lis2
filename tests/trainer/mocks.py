# tests/mocks.py
import torch
from torch import nn

from src.datasets.dataset import BaseDataset
from src.early_stop import EarlyStopping
from src.metrics import BaseMetric
from src.models.base_model import BaseModel
from src.optimizer import BaseOptimizer
from src.scheduler import BaseScheduler


class MockDataset(BaseDataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return self._create_sample(idx)

    def _create_sample(self, idx):
        return {
            "inputs": torch.randn(30, 30, dtype=torch.float32),
            "target": torch.rand(30, 30, dtype=torch.float32),
            "labelled": torch.ones(30, 30, dtype=torch.uint8),
        }


class MockModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Define a simple convolutional layer for segmentation
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)

    def _preprocess_forward(self, inputs, **kwargs):
        """
        Preprocess inputs by adding a channel dimension.
        Expected input shape: [batch_size, 3, 3]
        Output shape: [batch_size, 1, 3, 3]
        """
        inputs = inputs.unsqueeze(1)  # Add channel dimension
        return inputs

    def _core_forward(self, x):
        """
        Core forward pass through the convolutional layer.
        """
        x = self.conv1(x)
        x = torch.sigmoid(x)  # Ensure outputs are between 0 and 1
        x = x.squeeze(1)  # Remove channel dimension
        return x


class MockOptimizer(BaseOptimizer):
    def __init__(self, params):
        defaults = {"lr": self.lr}
        super().__init__(params, defaults)

    def step(self):
        pass

    def zero_grad(self):
        pass


class MockScheduler(BaseScheduler):
    def step(self):
        pass

    def state_dict(self):
        return {}


class MockEarlyStopping(EarlyStopping):
    def step(self, loss):
        return False

    def reset(self):
        pass


class MockMetrics(BaseMetric):
    def update(self, preds, targets, idx):
        self.averaging_coef = 1
        self.result = 0.5


# Replace external dependencies with mocks in your test module
