import abc
from torch import nn
from PNRIA.configs.config import TypedCustomizable
import torch

class BaseModel(abc.ABC, TypedCustomizable, nn.Module):

    def forward(self, *args, **kwargs):
        x = self._preprocess_forward(*args, **kwargs)
        x = self._core_forward(x)
        loss = self._postprocess_forward(x)
        return loss

    @abc.abstractmethod
    def _core_forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _preprocess_forward(self, *args, **kwargs):
        pass

    def _postprocess_forward(self, x) -> torch.Tensor:
        """
        This func should return the loss
        """
        return x
