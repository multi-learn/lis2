import abc
from torch import nn
from PNRIA.configs.config import TypedCustomizable


class BaseModel(abc.ABC, TypedCustomizable, nn.Module):

    def forward(self, *args, **kwargs):
        x = self._preprocess_forward(*args, **kwargs)
        x = self._core_forward(x)
        return x

    @abc.abstractmethod
    def _core_forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _preprocess_forward(self, *args, **kwargs):
        pass
