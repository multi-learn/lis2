import abc
import torch

from PNRIA.configs.config import TypedCustomizable, Customizable, Schema


class Encoder(TypedCustomizable, abc.ABC):

    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, positions):
        raise NotImplementedError("Subclasses must implement the forward method")


class VariableEncoding(Customizable):
    """
    Configuration for a single variable used in position encoding.

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

    config_schema = {
        "index": Schema(int),
        "expand_dims": Schema(int),
        "scale": Schema(float),
        "offset": Schema(float, default=0.0),
        "unsqueeze": Schema(bool),
        "angle": Schema(float, optional=True),
    }

    def __init__(self, *args, **kwargs):
        super(VariableEncoding, self).__init__(*args, **kwargs)


class PositionEncoding(Encoder):
    """
    Base class for position encodings.

    PositionEncoding subclasses are responsible for encoding positional information based on a set of `VariableEncoding` instances.

    Example configuration:
        {
            "type": "PositionEncoding",
            "vars_config": [
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

    config_schema = {
        "vars_config": Schema(list),
    }

    aliases = ["Encoding"]

    def __init__(self, *args, **kwargs):
        super(PositionEncoding, self).__init__(*args, **kwargs)
        self.vars = [VariableEncoding.from_config(v) for v in self.vars_config]
        self._sort_vars_by_index()

    def _sort_vars_by_index(self):
        """Sorts the list of variables by their index."""
        self.vars.sort(key=lambda var: var.index)


class SinPositionEncoding(PositionEncoding):
    """
    Sine-based position encoding.

    Encodes positional information using a sine function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "SinPositionEncoding",
            "vars_config": [
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
    """
    Linear position encoding.

    Encodes positional information using a linear function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "LinPositionEncoding",
            "vars_config": [
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
    """
    Symmetric position encoding.

    Encodes positional information using a symmetric function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "SymPositionEncoding",
            "vars_config": [
                {
                    "index": 0,
                    "expand_dims": 2,
                    "scale": 1.0,
                    "unsqueeze": True,
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


class IdentityPositionEncoding(Encoder):
    """
    Identity position encoding.

    Encodes positional information using the identity function, with support for additional transformations based on the configuration.

    Example configuration:
        {
            "type": "IdentityPositionEncoding",
        }
    """

    aliases = ["IdentityEncoding", "Identity"]

    def __init__(self, *args, **kwargs):
        super(IdentityPositionEncoding, self).__init__(*args, **kwargs)

    def forward(self, positions):
        return positions
