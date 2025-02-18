Datasets
========

.. currentmodule:: datasets

BaseDataset
-----------

.. automodule:: src.datasets.dataset.BaseDataset
   :members:
   :undoc-members:
   :show-inheritance:

FilamentsDataset
----------------

.. automodule:: src.datasets.dataset.FilamentsDataset
   :members:
   :undoc-members:
   :show-inheritance:

**Disclamer:** In most cases, when they are necessary (i.e ``k-folds``), ``fold_assignments`` and ``fold_list`` are computed automatically by the :ref:`controller` and don't have to be specified. 

The FilamentsDataset class handles the loading, preprocessing, and augmentation of an HDF5 dataset. Below is a structured breakdown of its internal process:

1. Initialization

    The dataset is loaded from the specified dataset_path, using the HDF5 format.
    The following dataset components are retrieved:

    - patches: Input data patches.
    - spines: Target labels.
    - labelled: Mask indicating labelled pixels.


    A set of parameters to encode is created based on the ``toEncode`` list (i.e. the position of the patch in our example).
    The create_mapping method is called to organize the dataset indices, i.e. to map with the fold assignment created by the :ref:`controller`.
    Data augmentation techniques are initialized based on the data_augmentations configuration.

2. Preconditions

    The preconditions method ensures consistency in dataset configuration:

    - If fold assignments are provided, both fold_assignments and fold_list must be defined.
    - If neither is provided, all dataset patches are used.

3. Dataset Mapping (create_mapping)

    This method organizes dataset indices based on:

    - Fold assignments (fold_assignments).
    - Stride (stride) for sampling.
    - The use_all_patches flag, which determines whether all patches are included.

    If fold_assignments exist:

    - It ensures only patches that have been assigned to folds are used.
    - If use_all_patches is False, only patches containing at least one labelled pixel are retained.

    This method is particularly useful because some patches may not be included, depending on ``fold_assignment``.

4. Sample Retrieval (__getitem__)

    The dataset retrieves samples using index mapping (dic_mapping).
    A patch, its corresponding label, and any encoded parameters are extracted.
    Data augmentation is applied if configured.
    The _create_sample method formats the data into tensors.

    .. note::
        This is the output that will be used by the trainer, so if you make a new ``dataset`` or a new ``trainer``, please make 
        sure that they correspond to one another.

5. Sample Creation (_create_sample)

    Converts the patch, spine labels, and labelled mask into PyTorch tensors.
    Ensures correct tensor formatting (C, H, W).
    Encodes additional parameters if specified.
    Validates that the target tensor is non-empty.

6. Logging and Debugging

    Logs whether fold assignments are used.
    Warns if no valid data is found for the given fold configuration.
    Ensures patches are properly formatted and contain valid labels.

Subtleties:

    - Fold-aware Data Loading: Ensures patches are correctly assigned to training, validation, and test folds.
    - Efficient Sampling: Uses stride-based mapping for scalable dataset handling.
    - Data Integrity Checks: Ensures no empty target tensors are included.
    - Augmentation Support: Applies configurable transformations to enhance training diversity.

.. autoclass:: src.datasets.fakeData3d.Fake3DDataset
   :members:
   :undoc-members:
   :show-inheritance: