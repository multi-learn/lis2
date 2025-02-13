Encoders
=============================

Encoders are responsible for encoding positional information before passing it to a model.
They transform raw position data into meaningful representations that can be used in deep learning architectures.

Available Encoders for position
-------------------------------

Several built-in encoders are provided:

- :class:`encoder.BaseEncoder` - Base class for all encoders.
- :class:`encoder.PositionEncoding` - Handles multiple variables for position encoding.
- :class:`encoder.SinPositionEncoding` - Uses sine-based encoding.
- :class:`encoder.LinPositionEncoding` - Uses linear scaling.
- :class:`encoder.IdentityPositionEncoding` - Directly passes input positions without transformation.

Configuration
-------------

Each encoder requires a configuration dictionary that defines how the positional encoding should behave.

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    name: "sin_encoder"
    type: "SinPositionEncoding"
    vars_config:
      - index: 0
        expand_dims: 2
        scale: 1.0
        offset: 0.0
        unsqueeze: True
        angle: 30.0

- **index**: Specifies which dimension of the input data should be encoded.
- **expand_dims**: Defines how the encoding should be expanded spatially.
- **scale**: Scaling factor applied before encoding.
- **offset**: Offset added to the input before applying transformations.
- **unsqueeze**: If `True`, expands dimensions to match the model’s expected input.
- **angle** (optional): Used in :class:`encoder.SinPositionEncoding` to adjust frequency.

Using an Encoder
----------------

Encoders can be instantiated and used as follows:

.. code-block:: python

    import torch
    from encoder import SinPositionEncoding

    positions = torch.randn(10, 5, 1)  # Batch of 10, 5 positional indices
    encoder_config = {
        "type": "SinPositionEncoding",
        "vars_config": [
            {"index": 0, "expand_dims": 2, "scale": 1.0, "offset": 0.0, "unsqueeze": True, "angle": 30.0}
        ]
    }

    encoder = BaseEncoder.from_config(encoder_config)
    encoded_positions = encoder(positions)
    print(encoded_positions.shape)  # Expected shape: (10, X, X), where X depends on `expand_dims`

Creating a Custom Encoder
-------------------------

You can create a custom encoder by subclassing :class:`encoder.PositionEncoding` and implementing `forward`:

.. code-block:: python

    from encoder import PositionEncoding
    import torch

    class CustomPositionEncoding(PositionEncoding):
        def forward(self, positions):
            encoded = []
            for v in self.vars:
                pe = torch.log1p(positions[:, v.index, 0] * v.scale + v.offset)
                if v.unsqueeze:
                    pe = torch.unsqueeze(
                        torch.unsqueeze(pe, dim=2).expand(pe.shape[0], v.expand_dims, v.expand_dims), dim=1
                    )
                encoded.append(pe)
            return torch.cat(encoded, dim=1)

    # Usage example
    custom_encoder_config = {
        "type": "CustomPositionEncoding",
        "vars_config": [
            {"index": 0, "expand_dims": 2, "scale": 2.0, "offset": 1.0, "unsqueeze": True}
        ]
    }

    custom_encoder = BaseEncoder.from_config(custom_encoder_config)
    test_positions = torch.randn(10, 5, 1)
    output = custom_encoder(test_positions)
    print(output.shape)
