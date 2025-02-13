Models Module
=============

This module provides a collection of deep learning models implemented in PyTorch. Each model is designed for different tasks, including segmentation, classification, and denoising.

.. currentmodule:: models

BaseModel Class
---------------

.. autoclass:: BaseModel
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseModel`` class serves as the foundation for all models, providing common functionalities such as initialization, weight loading, and saving.

BaseUNet Model
--------------

.. autoclass:: BaseUNet
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

CNN1D Model
-----------

.. autoclass:: CNN1D
   :members:
   :undoc-members:
   :show-inheritance:

The ``CNN1D`` model implements a 1D convolutional neural network architecture, suitable for tasks involving sequential or time-series data.

DNCNN Model
-----------

.. autoclass:: DnCNN
   :members:
   :undoc-members:
   :show-inheritance:

The ``DnCNN`` model is a denoising convolutional neural network, often used for image denoising applications.

UNet++ Model
------------

.. autoclass:: UNetPP
   :members:
   :undoc-members:
   :show-inheritance:

The ``UNetPP`` model extends the U-Net architecture with dense connections, improving performance for medical imaging and segmentation tasks.

VGG1D Model
-----------

.. autoclass:: VGG1D
   :members:
   :undoc-members:
   :show-inheritance:

The ``VGG1D`` model is a 1D adaptation of the VGG architecture, commonly used for time-series classification.

VNet Model
----------

.. autoclass:: VNet
   :members:
   :undoc-members:
   :show-inheritance:

The ``VNet`` model is a 3D segmentation network designed for medical imaging tasks, particularly volumetric segmentation.

Notes

- All models inherit from ``BaseModel`` and follow a modular design for easy extensibility.
