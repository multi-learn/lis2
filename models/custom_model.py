import abc

import torch
from configurable import TypedConfigurable
from torch import nn


class BaseModel(abc.ABC, TypedConfigurable, nn.Module):

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

    @classmethod
    def from_snapshot(cls, snapshot):
        """
        Load the model from a snapshot using a recursive search.
        """
        def find_keys(d, keys):
            found = {}
            if not isinstance(d, dict):
                return found

            for k, v in d.items():
                if k in keys:
                    found[k] = v
                elif isinstance(v, dict):
                    found.update(find_keys(v, keys))
            return found
        if not isinstance(snapshot, dict):
            try:

                snapshot = torch.load(snapshot, weights_only=False)
            except Exception as e:
                raise ValueError(f"Could not load snapshot: {e}")
        required_keys = {"MODEL_CONFIG", "MODEL_STATE"}
        results = find_keys(snapshot, required_keys)
        if not required_keys.issubset(results):
            raise ValueError(f"Snapshot is missing required keys: {required_keys - results.keys()}")
        model_config = results["MODEL_CONFIG"]
        model_state = results["MODEL_STATE"]
        model = cls.from_config(model_config)
        model.load_state_dict(model_state)
        return model

