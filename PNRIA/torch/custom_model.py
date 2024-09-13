import abc
from torch import nn
from PNRIA.configs.config import Configurable


class BaseModel(abc.ABC, Configurable, nn.Module):

    def forward(self, *args, **kwargs):
        x = self._preprocess_forward(*args, **kwargs)
        x = self._core_forward(x)
        return x

    @abc.abstractmethod
    def _core_forward(self, x):
        pass

    @abc.abstractmethod
    def _preprocess_forward(self, *args, **kwargs):
        pass
