Segmenter Module
================


This module provides the `Segmenter` class, which is designed to handle segmentation tasks using configurable models and datasets. It supports loading models from snapshots, processing data in batches, and saving the segmentation output to a FITS file.


Segmentation Process
--------------------

The `segment` method orchestrates the segmentation process. Here's a detailed explanation of its internal workings:

1. **Initialization:**
   - The method begins by setting the model to evaluation mode and moving it to the appropriate device (CPU or GPU).
   - A DataLoader is created for processing the dataset in batches.

2. **Batch Processing:**
   - For each batch, the method processes the data through the model to obtain segmented patches.
   - The segmented patches are then integrated into the overall segmentation map based on their positions.

3. **Handling Missing Data:**
   - If the `missing` flag is set, the method applies a missing data mask to the segmentation map.

4. **Post-processing:**
   - The segmentation map is normalized using a count map that tracks how many times each pixel was updated.
   - Morphological operations are applied to refine the segmentation map if segmentation is not disabled.

5. **Saving Output:**
   - The final segmentation map is saved to a FITS file if `save_output` is enabled.

**Subtleties:**

- **Device Management:** The segmenter automatically selects the device (CPU or GPU) based on availability.
- **Missing Data Handling:** The method can handle missing data by applying a mask during the segmentation process.
- **Morphological Operations:** Post-processing includes erosion and reconstruction to refine the segmentation map.

Example Usage
-------------

Here's an example of how to use the `Segmenter` class:

.. code-block:: python

    config = {
        "model_snapshot": "path/to/best.pt",
        "source": "path/to/data/spine_merged.fits",
        "dataset": {
            "type": "FilamentsDataset",
            "dataset_path": "path/to/data/test.h5",
            "learning_mode": "onevsall",
            "toEncode": ["positions"],
        },
        "batch_size": 16,
        "missing": False,
        "output_path": "segmentation_output.fits",
    }

    segmenter = Segmenter.from_config(config)
    segmenter.segment()


Segmenter
---------------

The ``Segmenter`` class is responsible for performing segmentation tasks using a configurable model and dataset. It supports handling missing data and disabling segmentation if needed.


.. autoclass:: lis2.segmenter.Segmenter
   :members:
   :undoc-members:
   :show-inheritance:
