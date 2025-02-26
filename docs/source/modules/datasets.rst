Datasets
========

.. currentmodule:: datasets

BaseDataset
-----------

.. automodule:: src.datasets.dataset.BaseDataset
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Zoo
-----------


FilamentsDataset
****************

.. automodule:: src.datasets.dataset.FilamentsDataset
   :members:
   :undoc-members:
   :show-inheritance:

**Disclamer:** In most cases, when they are necessary (i.e ``k-folds``), ``fold_assignments`` and ``fold_list`` are computed automatically by the :ref:`controller` and don't have to be specified. 

.. note::
     Torchvision Transforms and Custom Transforms are supported as ``Data Augmentations``. Please refer to :ref:`Data Augmentation` for more details about the data augmentations

The FilamentsDataset class handles the loading, preprocessing, and augmentation of an HDF5 dataset. Below is a structured breakdown of its internal process:

1. Initialization

    The dataset is loaded from the specified dataset_path, using the HDF5 format.
    The following dataset components are retrieved:

    - **patches**: Input data patches.
    - **spines**: Target labels.
    - **labelled**: Mask indicating labelled pixels.


    A set of parameters to encode is created based on the ``toEncode`` list (i.e. the ``position`` of the patch in our example).
    The ``create_mapping`` method is called to organize the dataset indices, i.e. to map with the ``fold assignment`` created by the :ref:`controller`.
    ``Data augmentation`` techniques are initialized based on the ``data_augmentations`` configuration.

2. Preconditions

    The ``preconditions`` method ensures consistency in dataset configuration:

    - If ``fold assignments`` are provided, both ``fold_assignments`` and ``fold_list`` must be defined.
    - If neither is provided, all dataset patches are used.

3. Dataset Mapping (create_mapping)

    This method organizes dataset indices based on:

    - **Fold assignments**.
    - **Stride** for sampling (i.e. if ``stride==2``, we take one out of two patches).
    - The **use_all_patches** flag, which determines whether all patches are included.

    If ``fold_assignments`` exists:

    - It ensures only patches that have been assigned to folds are used.
    - If ``use_all_patches`` is False, only patches containing at least one labelled pixel are retained.

    This method is particularly useful because some patches may not be included, depending on ``fold_assignment``.

4. Sample Retrieval (__getitem__)

    The dataset retrieves samples using index mapping (``dic_mapping``).
    A ``patch``, its corresponding ``label``, and any ``encoded parameters`` are extracted.
    ``Data augmentation`` is applied if configured.
    The _create_sample method formats the data into tensors.

    .. note::
        This is the output that will be used by the ``trainer``, so if you make a new ``dataset`` or a new ``trainer``, please make 
        sure that they correspond to one another.

5. Sample Creation (_create_sample)

    Converts the ``patch``, ``spine`` labels, and ``labelled`` mask into PyTorch tensors.
    Ensures correct tensor formatting ``(C, H, W)``.
    Encodes additional parameters if specified.
    Validates that the target tensor is non-empty.

6. Logging and Debugging

    Logs whether fold assignments are used.
    Warns if no valid data is found for the given fold configuration.
    Ensures patches are properly formatted and contain valid labels.

Subtleties:

    - **Fold-aware Data Loading**: Ensures patches are correctly assigned to training, validation, and test folds.
    - **Efficient Sampling**: Uses stride-based mapping for scalable dataset handling.
    - **Data Integrity Checks**: Ensures no empty target tensors are included.
    - **Augmentation Support**: Applies configurable transformations to enhance training diversity.

.. autoclass:: src.datasets.fakeData3d.Fake3DDataset
   :members:
   :undoc-members:
   :show-inheritance:

Add a Custom Dataset
--------------------

First, create a class that inherits from ``BaseDataset``, and create its ``config_schema``:

.. code-block:: python

    class CustomData(BaseDataset):
        """
        Your custom dataset

        It must implements __len__, _create_sample and __getitem__.
        """
        config_schema = {
            "dataset_path": Schema(Union[Path, str]),
            "data_augmentations": Schema(List[Config], optional=True),
            "toEncode": Schema(list, optional=True, default=[]),
            "stride": Schema(int, default=1),
            "fold_assignments": Schema(dict, optional=True),
            "fold_list": Schema(list, optional=True),
        }

Now you need to implement the methods :

.. code-block:: python

    data = # Load your data here

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self):
        """
        Important note : This is the output that will be used by the trainer. Make sure that they correspond
        """

        item1 = self.data["item1"]
        item2 = self.data["item2"]

        data_to_augment = {"item1": item1, "item2": item2}
        item1, item2 = self.data_augmentations.compute(data_to_augment)

        sample = self._create_sample(item1, item2)

        return sample

.. note::
    In ``getitem``, please make sure that the name of the variables in ``data_to_augment`` corresponds to what you specify 
    in ``keys_to_encode`` in ``data_augmentations``.

.. code-block:: python

    def _create_sample(self, item1, item2):

        sample = {
            "item1": item1,
            "item2": item2,
        }
        return sample