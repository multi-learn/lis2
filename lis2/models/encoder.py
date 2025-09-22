import abc

import torch
from configurable import TypedConfigurable, Configurable, Schema


class BaseEncoder(TypedConfigurable, abc.ABC):
    """
    Encoder base class for encoding mechanisms.

    This abstract class serves as the foundation for different position encoding implementations.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, positions):
        """Encodes the given positions."""
        raise NotImplementedError("Subclasses must implement the forward method")


class VariableEncoding(Configurable):
    """
    VariableEncoding for defining position encoding parameters.

    Configuration:
        - **index** (int): Index of the variable.
        - **expand_dims** (int): Number of dimensions to expand.
        - **scale** (float): Scaling factor for the variable.
        - **offset** (float): Offset value. Default is 0.0.
        - **unsqueeze** (bool): Whether to unsqueeze the dimensions.
        - **angle** (float, optional): Angle parameter.

    Example Configuration (YAML):
        .. code-block:: yaml

            index: 0
            expand_dims: 2
            scale: 1.0
            offset: 0.0
            unsqueeze: True
            angle: 30.0
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
        super().__init__(*args, **kwargs)


class PositionEncoding(BaseEncoder):
    """
    PositionEncoding for encoding positional information.

    This is a base class for encoding positions using different transformation functions.
    It processes input positions using a set of `VariableEncoding` configurations, which define
    how each variable contributes to the encoding.

    Configuration:
        - **name** (str): The name of the encoding instance.
        - **vars_config** (list): A list of `VariableEncoding` configurations defining the encoding parameters.

    Example Configuration (YAML):
        .. code-block:: yaml

            name: "position_encoding"
            vars_config:
              - index: 0
                expand_dims: 2
                scale: 1.0
                offset: 0.0
                unsqueeze: True
                angle: 30.0

    Aliases:
        Encoding
    """

    config_schema = {
        "vars_config": Schema(list),
    }

    aliases = ["Encoding"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vars = [VariableEncoding.from_config(v) for v in self.vars_config]
        self._sort_vars_by_index()

    def _sort_vars_by_index(self):
        """Sorts the list of variables by their index."""
        self.vars.sort(key=lambda var: var.index)


class SinPositionEncoding(PositionEncoding):
    """
    SinPositionEncoding for sine-based position encoding.

    Configuration:
        Inherits from PositionEncoding.

    Aliases:
        SinEncoding, Sin
    """

    aliases = ["SinEncoding", "Sin"]

    def forward(self, positions):
        encoded = []
        for v in self.vars:
            pe = torch.abs(
                torch.cos(
                    ((v.scale - positions[:, v.index, 0] * v.offset + v.angle))
                    * torch.pi
                    / v.angle
                    * 2
                )
            )
            if v.unsqueeze:
                pe = torch.unsqueeze(
                    torch.unsqueeze(pe, dim=2).expand(
                        pe.shape[0], v.expand_dims, v.expand_dims
                    ),
                    dim=1,
                )
            encoded.append(pe)
        return torch.cat(encoded, dim=1)


class LinPositionEncoding(PositionEncoding):
    """
    LinPositionEncoding for linear position encoding.

    Configuration:
        Inherits from PositionEncoding.

    Aliases:
        LinEncoding, Lin
    """

    aliases = ["LinEncoding", "Lin"]

    def forward(self, positions):
        encoded = []
        for v in self.vars:
            pe = (positions[:, v.index, 0]) / v.scale
            if v.unsqueeze:
                pe = torch.unsqueeze(
                    torch.unsqueeze(pe, dim=2).expand(
                        pe.shape[0], v.expand_dims, v.expand_dims
                    ),
                    dim=1,
                )
            encoded.append(pe)
        return torch.cat(encoded, dim=1)


class IdentityPositionEncoding(BaseEncoder):
    """
    IdentityPositionEncoding for identity transformation of positional information.

    Configuration:
        Inherits from Encoder.

    Aliases:
        IdentityEncoding, Identity
    """

    aliases = ["IdentityEncoding", "Identity"]

    def forward(self, positions):
        return positions
