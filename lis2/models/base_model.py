import abc
from typing import Any, Dict, Type

import torch
from configurable import TypedConfigurable
from torch import nn


class BaseModel(abc.ABC, TypedConfigurable, nn.Module):
    """
    BaseModel for creating neural network models with preprocessing, core processing, and postprocessing steps.

    This class serves as a template for creating neural network models following the **Template Method** design pattern.
    It defines a structure that includes preprocessing of input data, core processing where the main model logic resides,
    and postprocessing to compute the final output or loss. Subclasses must implement the abstract methods to define
    specific behavior for these steps.

    **Important note**:

    - The `forward` method orchestrates the sequence of operations and should not be overridden. Instead, subclasses should implement the `core_forward`, `preprocess_forward`, and `postprocess_forward` methods to define their specific processing logic.
    - The arguments `*args: Any, **kwargs: Any` in the `forward` method are dependent on the dataset used, as defined in :ref:`BaseDataset`.

    Configuration:
        - **name** (str): The name of the model.

    Example Configuration for from_snapshot (YAML):
        .. code-block:: yaml

            name: "example_model"
    """

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the model, which includes preprocessing, core processing, and postprocessing.

        This method orchestrates the sequence of operations for the forward pass:

            1. Preprocess the input data using :meth:`preprocess_forward`.
            2. Apply the core model logic using :meth:`core_forward`.
            3. Postprocess the output to compute the loss using :meth:`postprocess_forward`.

        **Note**: This method should not be overridden in subclasses. Instead, implement the abstract methods
        :meth:`core_forward`, :meth:`preprocess_forward`, and :meth:`postprocess_forward`.
        """
        x = self.preprocess_forward(*args, **kwargs)
        x = self.core_forward(x)
        loss = self.postprocess_forward(x)
        return loss

    @abc.abstractmethod
    def core_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for the core forward pass logic. Must be implemented by subclasses.

        This method contains the main logic of the model, such as the neural network layers
        and their interactions. It processes the preprocessed input data and produces an output
        that will be further processed to compute the loss. The input to this method is the
        output from the `preprocess_forward` method.

        Args:
            *args: Variable length arguments for the core forward pass. These arguments are
                   dependent on the output of the `preprocess_forward` method.
            **kwargs: Arbitrary keyword arguments for the core forward pass.

        Returns:
            Any: The processed output from the core forward pass, which will be used in the
                 `postprocess_forward` method to compute the loss.
        """
        pass

    @abc.abstractmethod
    def preprocess_forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method for preprocessing input data before the core forward pass. Must be implemented by subclasses.

        This method handles any necessary preprocessing of the input data, such as normalization,
        reshaping, or any other transformations required before feeding the data into the core model logic.
        The input to this method is dependent on the output of the `__getitem__` method of the dataset.

        Args:
            *args: Variable length arguments for preprocessing. These arguments are dependent on
                   the dataset's `__getitem__` method.
            **kwargs: Arbitrary keyword arguments for preprocessing.

        Returns:
            Any: The preprocessed input data, which will be passed to the `core_forward` method.
        """
        pass

    @abc.abstractmethod
    def postprocess_forward(self, x: Any) -> torch.Tensor:
        """
        Postprocess the output of the core forward pass to compute the loss.

        This method takes the output from the core forward pass and computes the final loss value.
        It can include operations such as applying a loss function or any other final transformations.
        The input to this method is the output from the `core_forward` method.

        Args:
            x (Any): The output from the core forward pass.

        Returns:
            torch.Tensor: The computed loss value.
        """
        return x

    @classmethod
    def from_snapshot(cls: Type["BaseModel"], snapshot: Dict[str, Any]) -> "BaseModel":
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
            raise ValueError(
                f"Snapshot is missing required keys: {required_keys - results.keys()}"
            )

        model_config = results["MODEL_CONFIG"]
        model_state = results["MODEL_STATE"]
        model = cls.from_config(model_config)
        model.load_state_dict(model_state)

        return model
