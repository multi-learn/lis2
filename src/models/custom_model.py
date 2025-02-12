import abc
from typing import Any, Dict, Type

import torch
from configurable import TypedConfigurable
from torch import nn


class BaseModel(abc.ABC, TypedConfigurable, nn.Module):
    """
    Base class for models, integrating abstract methods for core processing and pre/post-processing steps.

    This class serves as a template for creating neural network models. It defines a structure
    that includes preprocessing of input data, core processing where the main model logic resides,
    and postprocessing to compute the final output or loss. Subclasses must implement the
    abstract methods to define specific behavior for these steps.
    """

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the model, which includes preprocessing, core processing, and postprocessing.

        This method orchestrates the sequence of operations for the forward pass:
        1. Preprocess the input data.
        2. Apply the core model logic.
        3. Postprocess the output to compute the loss.
        """
        x = self._preprocess_forward(*args, **kwargs)
        x = self._core_forward(x)
        loss = self._postprocess_forward(x)
        return loss

    @abc.abstractmethod
    def _core_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for the core forward pass logic. Must be implemented by subclasses.

        This method should contain the main logic of the model, such as the neural network layers
        and their interactions. It processes the preprocessed input data and produces an output
        that will be further processed to compute the loss.
        """
        pass

    @abc.abstractmethod
    def _preprocess_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for preprocessing input data before the core forward pass. Must be implemented by subclasses.

        This method should handle any necessary preprocessing of the input data, such as normalization,
        reshaping, or any other transformations required before feeding the data into the core model logic.
        """
        pass

    def _postprocess_forward(self, x: Any) -> torch.Tensor:
        """
        Postprocess the output of the core forward pass to compute the loss.

        This method takes the output from the core forward pass and computes the final loss value.
        It can include operations such as applying a loss function or any other final transformations.
        """
        return x

    @classmethod
    def from_snapshot(cls: Type['BaseModel'], snapshot: Dict[str, Any]) -> 'BaseModel':
        """
        Load the model from a snapshot.

        This method loads a model from a snapshot, which can be a dictionary containing the model's
        configuration and state, or a path to a file containing this information. It recursively
        searches for required keys in the snapshot dictionary to extract the model configuration
        and state, then initializes the model and loads its state.

        Args:
            snapshot (Dict[str, Any]): A dictionary containing the snapshot or a path to the snapshot file.

        Returns:
            BaseModel: An instance of the model loaded from the snapshot.

        Raises:
            ValueError: If the snapshot is missing required keys or cannot be loaded.
        """

        def find_keys(d: Dict[str, Any], keys: set) -> Dict[str, Any]:
            """
            Recursively search for specific keys in a nested dictionary.
            """
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
