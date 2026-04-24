Models
======

This module provides a collection of deep learning models implemented in PyTorch. Each model is designed for different tasks, including segmentation, classification, and denoising.

.. currentmodule:: models

.. _adding_model:

Adding a Custom Model
---------------------

This tutorial walks you through the process of adding a custom model to the existing codebase. We will define a new model by extending the ``BaseModel`` class, configure it, and update the ``__init__.py`` file to ensure the model is accessible.

Prerequisites
*************

Before you start, ensure you have the following:

- Basic understanding of Python and PyTorch.
- Familiarity with the existing codebase and its structure.
- Access to the necessary datasets and configuration files.

Step 1: Define the Custom Model Class
*************************************

To create a custom model, you need to define a new class that extends the ``BaseModel`` class. This class should implement the abstract methods: ``core_forward``, ``preprocess_forward``, and ``postprocess_forward``.

Here is an example of how to define a custom model class:

.. code-block:: python

    from models import BaseModel, nn
    from torch import nn
    from configurable import TypedConfigurable, Schema

    class CustomModel(BaseModel):
        """
        CustomModel is an example model extending BaseModel.

        This model implements specific logic for preprocessing, core processing,
        and postprocessing steps. It uses a configurable number of layers.

        Configuration:
            - name (str): The name of the model.
            - type (str): The type of the model.
            - n_layer (int): The number of layers in the model. Default is 5.
        """

        config_schema = {
            "n_layer": Schema(int, default=5)
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Initialize layers based on the configuration
            self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(self.n_layer)])

        def core_forward(self, x: torch.Tensor) -> torch.Tensor:
            # Example core logic using multiple layers
            for layer in self.layers:
                x = layer(x)
            return x

        def preprocess_forward(self, *args, **kwargs) -> torch.Tensor:
            # Example preprocessing logic
            data, _ = args[0]  # Assuming args[0] is a tuple from the dataset
            return data.float() / 255.0  # Normalize data

        def postprocess_forward(self, x: torch.Tensor) -> torch.Tensor:
            # Example postprocessing logic
            loss = torch.mean(x)  # Example loss computation
            return loss


Step 2: Configure the Custom Model
**********************************

After defining the custom model class, you need to configure it using a YAML configuration file. Here is an example configuration ``tuto_config.yml``:

.. code-block:: yaml

    name: "custom_model"
    type: "CustomModel"
    n_layer: 10


Make sure to place this configuration file in the appropriate directory and reference it when initializing the model.

Step 3: Update the __init__.py File
***********************************

To ensure that your custom model is accessible, you need to update the ``__init__.py`` file in the appropriate module directory. This file should import your custom model class so that it can be easily accessed by other parts of the codebase.

Here is an example of how to update the ``__init__.py`` file:

.. code-block:: python

    from .custom_model import CustomModel

    __all__ = [
        'BaseModel',
        'CustomModel',  # Add your custom model here
        # Other models and classes
    ]

This ensures that ``CustomModel`` is imported when the module is imported, making it accessible for use in other parts of the application.

Step 4: Initialize and Use the Custom Model
*******************************************

To initialize and use the custom model, you can use the following code:

.. code-block:: python

    from models import CustomModel

    model = CustomModel.from_config("/tuto_config.yml")
    print(model)

Conclusion
**********

You have now successfully added a custom model to the codebase by extending the ``BaseModel`` class, configuring it properly, and updating the ``__init__.py`` file. You can further customize the model by modifying the logic in the ``core_forward``, ``preprocess_forward``, and ``postprocess_forward`` methods according to your needs.

For more detailed information on each component, refer to the respective sections in the documentation:

- :ref:`BaseModel<BaseModel Class>`
- :ref:`BaseDataset`


BaseModel Class
---------------

.. autoclass:: lis2.models.BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseModel`` class serves as the foundation for all models, providing common functionalities such as initialization, weight loading, and saving.

Models Zoo
----------


UNet Model
**********

.. autoclass:: lis2.models.BaseUNet
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseUNet`` class implements a generic U-Net architecture, which is widely used for image segmentation tasks. It supports customizable depth and feature sizes.

Example usage:

.. code-block:: python

    from models.base_unet import BaseUNet

    config = {
        "name": "example_unet",
        "in_channels": 3,
        "out_channels": 1,
        "features": 64,
        "num_blocks": 4,
        "dim": 2
    }

    model = BaseUNet.from_config(config)
    print(model)

Encoder Position Enum
^^^^^^^^^^^^^^^^^^^^^

The ``encoder_pos`` enumeration specifies the position of the encoder in the U-Net architecture. The possible values are:

- ``BEFORE``: The encoder is placed before the encoding process begins. This configuration concatenates the position encoding with the input tensor at the start of the forward pass.

- ``MIDDLE``: The encoder is placed after the bottleneck layer. This configuration concatenates the position encoding with the tensor immediately after the bottleneck, influencing the upsampling path of the U-Net.

- ``AFTER``: The encoder is placed after the decoding process is complete. This configuration concatenates the position encoding with the output tensor at the end of the forward pass.

These positions determine when the position encoding is integrated into the U-Net model's forward pass, affecting how spatial information is utilized throughout the network.


CNN1D Model
***********

.. autoclass:: lis2.models.CNN1D
   :members:
   :undoc-members:
   :show-inheritance:

The ``CNN1D`` model implements a 1D convolutional neural network architecture, suitable for tasks involving sequential or time-series data.

DNCNN Model
***********

.. autoclass:: lis2.models.DnCNN
   :members:
   :undoc-members:
   :show-inheritance:

The ``DnCNN`` model is a denoising convolutional neural network, often used for image denoising applications.

Unet++ Model
************

.. autoclass:: lis2.models.UNetPP
   :members:
   :undoc-members:
   :show-inheritance:

The ``UNetPP`` model extends the U-Net architecture with dense connections, improving performance for medical imaging and segmentation tasks.

VGG1D Model
***********

.. autoclass:: lis2.models.VGG1D
   :members:
   :undoc-members:
   :show-inheritance:

The ``VGG1D`` model is a 1D adaptation of the VGG architecture, commonly used for time-series classification.

VNet Model
**********

.. autoclass:: lis2.models.VNet
   :members:
   :undoc-members:
   :show-inheritance:

The ``VNet`` model is a 3D segmentation network designed for medical imaging tasks, particularly volumetric segmentation.

Notes

- All models inherit from ``BaseModel`` and follow a modular design for easy extensibility.

Position Encoders
-----------------


Encoders are responsible for encoding positional information before passing it to a model.
They transform raw position data into meaningful representations that can be used in deep learning architectures.

Available Encoders for position
*******************************

Several built-in encoders are provided:

- :class:`encoder.BaseEncoder` - Base class for all encoders.
- :class:`encoder.PositionEncoding` - Handles multiple variables for position encoding.
- :class:`encoder.SinPositionEncoding` - Uses sine-based encoding.
- :class:`encoder.LinPositionEncoding` - Uses linear scaling.
- :class:`encoder.IdentityPositionEncoding` - Directly passes input positions without transformation.

Configuration
*************

Each encoder requires a configuration dictionary that defines how the positional encoding should behave.

Example Configuration
*********************

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
****************

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
*************************

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

Encoder Position Zoo
********************

BaseEncoder
^^^^^^^^^^^

.. autoclass:: lis2.models.BaseEncoder
   :members:
   :undoc-members:
   :show-inheritance:

VariableEncoding
^^^^^^^^^^^^^^^^

.. autoclass:: lis2.models.VariableEncoding
   :members:
   :undoc-members:
   :show-inheritance:

PositionEncoding
^^^^^^^^^^^^^^^^

.. autoclass:: lis2.models.PositionEncoding
   :members:
   :undoc-members:
   :show-inheritance:

SinPositionEncoding
^^^^^^^^^^^^^^^^^^^

.. autoclass:: lis2.models.SinPositionEncoding
   :members:
   :undoc-members:
   :show-inheritance:

LinPositionEncoding
^^^^^^^^^^^^^^^^^^^

.. autoclass:: lis2.models.LinPositionEncoding
   :members:
   :undoc-members:
   :show-inheritance:

IdentityPositionEncoding
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: lis2.models.IdentityPositionEncoding
   :members:
   :undoc-members:
   :show-inheritance:
