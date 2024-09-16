import abc
import torch
from PNRIA.configs.config import Configurable


class Encoder(abc.ABC, Configurable):
    """Abstract base class for encoders.

    Encoders are responsible for encoding positional information into a format suitable for further processing.

    Attributes:
        required_keys (list): List of keys that must be present in the configuration.

    Methods:
        __call__(*args, **kwargs): Calls the `forward` method to process input data.
        forward(positions): Abstract method that must be implemented by subclasses to perform encoding.

    Example configuration:
        {
            "type": "EncoderType",
            "other_config_key": "value"
        }

    How to create a custom encoder:
        To create a custom encoder, you need to follow these steps:

        1. **Subclass the `Encoder` Class:**
           Define a new class that inherits from `Encoder`. This new class will be your custom encoder.

        2. **Implement the `forward` Method:**
           Provide a concrete implementation for the `forward` method. This method should define how the positional information is processed and encoded.

        3. **Define Required Configuration Keys:**
           Specify the required configuration keys in the `required_keys` attribute. These keys are necessary for initializing your custom encoder.

        4. **Optionally, Define Aliases:**
           Define a list of aliases for your encoder in the `aliases` attribute. This is useful if you want to support different names for the same type of encoder.

        5. **Create and Configure the Encoder:**
           Use your custom encoder by creating an instance of it and passing the necessary configuration.

        Example of a custom encoder:

        ```python
        class CustomEncoder(Encoder):

            required_keys = ["custom_param"]

            def __init__(self, *args, **kwargs):
                super(CustomEncoder, self).__init__(*args, **kwargs)
                # Initialize custom parameters here, but they automatically create by the Configurable class
                # self.custom_param = self.config["custom_param"]

            def forward(self, positions):
                # Implement your encoding logic here
                encoded = positions * self.custom_param  # Example operation
                return encoded

            aliases = ["CustomEncoding"]
        ```

        Example configuration for `CustomEncoder`:

        ```python
        {
            "type": "CustomEncoder",
            "custom_param": 2.0
        }
        ```

        In this example, the `CustomEncoder` multiplies the input positional data by `custom_param`. You should replace this logic with the actual encoding logic required for your application.
    """

    required_keys = ["type"]

    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, positions):
        raise NotImplementedError("Subclasses must implement the forward method")



class VariableEncoding(Configurable):
    """Configuration for a single variable used in position encoding.

    Each `VariableEncoding` instance represents a specific variable that contributes to the positional encoding.

    Example configuration:
        {
            "index": 0,
            "expand_dims": 2,
            "scale": 1.0,
            "offset": 0.0,
            "unsqueeze": True,
            "angle": 30.0
        }
    """

    required_keys = ["index", "expand_dims", "scale", "unsqueeze",]

    def __init__(self, *args, **kwargs):
        super(VariableEncoding, self).__init__(*args, **kwargs)

class PositionEncoding(Encoder):
    """Base class for position encodings.

    PositionEncoding subclasses are responsible for encoding positional information based on a set of `VariableEncoding` instances.

    Example configuration:
        {
            "type": "PositionEncoding",
            "var": [
                {
                    "index": 0,
                    "expand_dims": 2,
                    "scale": 1.0,
                    "offset": 0.0,
                    "unsqueeze": True,
                    "angle": 30.0
                }
            ]
        }
    """

    required_keys = ["vars_config"]

    aliases = ["Encoding"]

    def __init__(self, *args, **kwargs):
        super(PositionEncoding, self).__init__(*args, **kwargs)
        self.vars = [VariableEncoding.from_config(v) for v in self.vars_config]
        self._sort_vars_by_index()

    def _sort_vars_by_index(self):
        """Sorts the list of variables by their index."""
        self.vars.sort(key=lambda var: var.index)


class SinPositionEncoding(PositionEncoding):
    """Sine-based position encoding.

    Encodes positional information using a sine function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "SinPositionEncoding",
            "var": [
                {
                    "index": 0,
                    "expand_dims": 2,
                    "scale": 1.0,
                    "offset": 0.0,
                    "unsqueeze": True,
                    "angle": 30.0
                }
            ]
        }
    """

    aliases = ["SinEncoding", "Sin"]

    def __init__(self, *args, **kwargs):
        super(SinPositionEncoding, self).__init__(*args, **kwargs)

    def forward(self, positions):
        encoded = []
        for v in self.vars:
            pe = torch.abs(torch.cos(
                ((v.scale - positions[:, v.index, 0] * v.scale + v.angle) + v.offset) * torch.pi / v.angle * 2))
            if v.unsqueeze:
                pe = torch.unsqueeze(
                    torch.unsqueeze(pe, dim=2).expand(pe.shape[0], v.expand_dims, v.expand_dims), dim=1)
            encoded.append(pe)
        return torch.cat(encoded, dim=1)


class LinPositionEncoding(PositionEncoding):
    """Linear position encoding.

    Encodes positional information using a linear function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "LinPositionEncoding",
            "var": [
                {
                    "index": 0,
                    "expand_dims": 2,
                    "scale": 1.0,
                    "offset": 0.0,
                    "unsqueeze": True,
                    "angle": 30.0
                }
            ]
        }
    """

    aliases = ["LinEncoding", "Lin"]

    def __init__(self, *args, **kwargs):
        super(LinPositionEncoding, self).__init__(*args, **kwargs)

    def forward(self, positions):
        encoded = []
        for v in self.vars:
            pe = positions[:, v.index, 0] / (v.scale * 2)
            if v.unsqueeze:
                pe = torch.unsqueeze(
                    torch.unsqueeze(pe, dim=2).expand(pe.shape[0], v.expand_dims, v.expand_dims), dim=1)
            encoded.append(pe)
        return torch.cat(encoded, dim=1)


class SymPositionEncoding(PositionEncoding):
    """Symmetric position encoding.

    Encodes positional information using a symmetric function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "SymPositionEncoding",
            "var": [
                {
                    "index": 0,
                    "expand_dims": 2,
                    "scale": 1.0,
                    "offset": 0.0,
                    "unsqueeze": True,
                    "angle": 30.0
                }
            ]
        }
    """

    aliases = ["SymEncoding", "Sym"]

    def __init__(self, *args, **kwargs):
        super(SymPositionEncoding, self).__init__(*args, **kwargs)

    def forward(self, positions):
        encoded = []
        for v in self.vars:
            pe = torch.abs((positions[:, v.index, 0] - v.scale) / v.scale)
            if v.unsqueeze:
                pe = torch.unsqueeze(
                    torch.unsqueeze(pe, dim=2).expand(pe.shape[0], v.expand_dims, v.expand_dims), dim=1)
            encoded.append(pe)
        return torch.cat(encoded, dim=1)
