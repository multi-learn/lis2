Data Augmentation
=================

This module provides a framework for integrating **Torchvision Transforms** with a configurable schema, allowing dynamic configuration and extension of transforms. It also enables the addition of custom data augmentations by inheriting from the ``BaseDataAugmentation`` class.

For more information about Torchvision Transforms, please refer to the `Torchvision Transforms v2 documentation <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.html>`_.

Overview
--------

The module is designed to be used within a dataset, where the data augmentation configuration becomes part of the dataset configuration. Each augmentation is applied sequentially according to the order specified in the configuration list.

.. note::
    **Important: Management of ``ToTensor``**

        - The configuration should include a ``ToTensor`` augmentation to ensure that the final output is a Tensor.
        - If no ``ToTensor`` augmentation is provided by the user, a default ``ToTensor`` transform is automatically appended to the configuration.
        - Although there is no strict constraint on the order of augmentations, care must be taken when adding new augmentations. The placement of the ``ToTensor`` transform should be handled similarly to a Torch Compose (see: `torchvision.transforms.Compose <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.html#torchvision.transforms.v2.Compose>`_) to avoid type errors.
        - **Note:** The input to the augmentation pipeline is expected to be NumPy arrays, while the final output must be a ``torch.Tensor``.

Usage
-----

This module is intended to be used as part of a dataset. Below is an example configuration:

.. code-block:: python

    data_augmentations = [
        {"type": "ToTensor", "dType": "float32"},
        {"type": "CenterCrop", "name": "center_crop", "size": 10},
        {"type": "NoiseDataAugmentation", "name": "input", "keys_to_augment": ["patch"], "noise_var": 0.1}
    ]

Key Points:
- **Sequential Application:** Each augmentation is applied sequentially in the order provided.
- **Mandatory Parameters:** Parameters such as ``name`` (for custom augmentations) and ``keys_to_augment`` may be required to differentiate augmentations and selectively apply transformations.
- **Selective Augmentation:** Use the parameter ``keys_to_augment`` to specify which data elements should be augmented. Leaving it empty applies the augmentation to all keys.

DataAugmentations Class
-----------------------

The main class responsible for applying the augmentations is ``DataAugmentations``. Its key functionalities include:

- **Configuration Validation:** Ensures that the provided configuration is a list.
- **Default Addition of ``ToTensor``:** If the configuration does not include a ``ToTensor`` augmentation, one is automatically appended.
- **Sequential Processing:** Iterates through the list of configured augmentations and applies each one to the input data.
- **Final Type Check:** After applying all augmentations, a type check is performed to ensure that each data element is a ``torch.Tensor`` (or a list of ``torch.Tensor`` objects). If not, an error is raised indicating a potential misplacement of the ``ToTensor`` transform.

.. autoclass:: src.datasets.data_augmentation.DataAugmentations
   :members:
   :undoc-members:
   :show-inheritance:

Augmentation Zoo
----------------

Custom Augmentations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: src.datasets.data_augmentation.NoiseDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.datasets.torch_data_augmentation.ToTensor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.datasets.torch_data_augmentation.RandomRotation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.datasets.torch_data_augmentation.RandomHorizontalFlip
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.datasets.torch_data_augmentation.RandomVerticalFlip
   :members:
   :undoc-members:
   :show-inheritance:

Available Torchvision Augmentations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following augmentations from ``torchvision.transforms.v2`` are available (non-excluded):

- `AugMix <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.AugMix.html>`_ (alias: augmix)
- `AutoAugment <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.AutoAugment.html>`_ (alias: autoaugment)
- `CenterCrop <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.CenterCrop.html>`_ (alias: centercrop)
- `ClampBoundingBoxes <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ClampBoundingBoxes.html>`_ (alias: clampboundingboxes)
- `ColorJitter <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ColorJitter.html>`_ (alias: colorjitter)
- `ConvertBoundingBoxFormat <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ConvertBoundingBoxFormat.html>`_ (alias: convertboundingboxformat)
- `ConvertImageDtype <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ConvertImageDtype.html>`_ (alias: convertimagedtype)
- `CutMix <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.CutMix.html>`_ (alias: cutmix)
- `ElasticTransform <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ElasticTransform.html>`_ (alias: elastictransform)
- `FiveCrop <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.FiveCrop.html>`_ (alias: fivecrop)
- `GaussianBlur <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.GaussianBlur.html>`_ (alias: gaussianblur)
- `GaussianNoise <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.GaussianNoise.html>`_ (alias: gaussiannoise)
- `Grayscale <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Grayscale.html>`_ (alias: grayscale)
- `Identity <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Identity.html>`_ (alias: identity)
- `JPEG <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.JPEG.html>`_ (alias: jpeg)
- `LinearTransformation <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.LinearTransformation.html>`_ (alias: lineartransformation)
- `MixUp <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.MixUp.html>`_ (alias: mixup)
- `Normalize <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Normalize.html>`_ (alias: normalize)
- `PILToTensor <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.PILToTensor.html>`_ (alias: piltotensor)
- `Pad <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Pad.html>`_ (alias: pad)
- `RGB <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RGB.html>`_ (alias: rgb)
- `Resize <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Resize.html>`_ (alias: resize)
- `ScaleJitter <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ScaleJitter.html>`_ (alias: scalejitter)
- `TenCrop <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.TenCrop.html>`_ (alias: tencrop)
- `ToDtype <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ToDtype.html>`_ (alias: todtype)
- `ToImage <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ToImage.html>`_ (alias: toimage)
- `ToPILImage <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ToPILImage.html>`_ (alias: topilimage)
- `ToPureTensor <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.ToPureTensor.html>`_ (alias: topuretensor)
- `Transform <https://pytorch.org/vision/main/generated/torchvision.transforms.v2.Transform.html>`_ (alias: transform)


.. _adding_augment:

Adding Custom Data Augmentation
-------------------------------

The framework allows you to easily extend its capabilities by creating your own custom augmentation. To do so, simply extend one of the abstract base classes provided below.

For a generic augmentation, extend ``BaseDataAugmentation``. If your augmentation needs to target specific keys within a dataset, extend ``BaseDataAugmentationWithKeys``. When using ``BaseDataAugmentationWithKeys``, the transformation will only be applied to the keys specified in the ``keys_to_augment`` configuration. If this list is empty, the augmentation is applied to all keys.

BaseDataAugmentation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.datasets.data_augmentation.BaseDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

BaseDataAugmentationWithKeys
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: src.datasets.data_augmentation.BaseDataAugmentationWithKeys
   :members:
   :undoc-members:
   :show-inheritance:
