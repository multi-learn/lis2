Controller
==============

.. currentmodule:: Controller

This module provides the ``Controller`` class, which is responsible for assigning the data to the right set (i.e. train, valid, test). If the configuration is a k-fold, then the controller role is to assign each patch to an area, and then each area to a fold.

Controller Class
-------------

The ``Controller`` classes are responsible for managing the complete lifecycle of fold creation, assignation of the data to these folds and split these folds into train-valid-test.

.. autoclass:: src.controller.FoldsController
   :members:
   :undoc-members:
   :show-inheritance:


Controller Process
------------------

The Controller orchestrates the process of generating k-fold splits for dataset training, validation, and testing, with an emphasis on patch-level organization and area-based grouping. Below is a detailed breakdown of its internal workings. Currently, two strategies (i.e. random or naive)
are implemented, and they both follow this framework:

1. **Initialization**

    The controller initializes by loading the dataset from the specified dataset_path.
    It sets up the fold indices path (indices_path) and determines whether to load precomputed indices or generate new ones.
    The k-fold splits are generated using generate_kfold_splits, considering the training ratio (k_train).
    The _create_folds method is invoked to distribute patches into folds based on their spatial organization.

2. **Preconditions**

    The preconditions method ensures the validity of k-fold parameters.
    It checks that (k * k_train) % 2 == 0, ensuring an even distribution of training data.

3. **Fold Creation Process (_create_folds)**

    This method is responsible for distributing dataset patches into k-folds.
    The patches are grouped into areas based on their coordinates within the dataset.
    If an indices file exists, the method loads precomputed fold assignments.
    If no precomputed indices exist, it dynamically computes area groups and saves them if save_indices is enabled.

4. **Area Grouping Strategy**

    Each patch is assigned to a specific area based on its top-left corner coordinates.
    The area coordinates are determined by the area_size parameter, while overlap is accounted for to avoid boundary issues.
    Patches that fall outside their assigned area due to overlap constraints are excluded.

5. **Assigning Areas to Folds**

    The controller distributes areas to folds using the defined strategy (see subtleties)

6. **Loading and Saving Fold Assignments**

    If save_indices is True, the computed fold assignments are saved to indices_path for future use.
    If an existing indices file is found, it is used instead of recomputing fold assignments.

7. **Logging and Debugging**

    The controller logs key steps, such as:
    - Loading precomputed indices (if available).
    - Assigning patches to areas and folds.
    - Saving computed fold indices.
    This helps with debugging and reproducibility.

**Subtleties**:

- Area-based Patch Assignment: Ensures that patches close together remain in the same fold, preventing data leakage.
- Implemented strategies: 
    - RandomController (round-robin manner) iterates through all computed area groups. Each area is assigned sequentially to one of the k folds. This ensures an even and systematic distribution of patches across training, validation, and testing sets.
    - NaiveController divides the images into k equal parts and assign each part to a fold.
- Reproducibility: Precomputed indices allow for consistent fold assignments across multiple runs.
- Scalability: Works efficiently with large datasets by structuring patches into manageable area groups.

Utils : Split generation
------------------------

A method called ``generate_kfold_splits`` is provided. It takes as input ``k``, the number of folds, and ``k_train``, which is the percentage of folds to put in the training set. Some edges cases are not handled :

- k must be pair or equal to one
- k - (k * k_train) must be equal to 2 or 4.
- k_train must be even

Example of usage and output for ``k=10`` and ``k_train = 0.60``:

.. code-block:: python

    splits = generate_kfold_splits(10, 0.60)
    splits
    [
        ([4, 5, 6, 7, 8, 9], [0, 1], [2, 3]),
        ([0, 5, 6, 7, 8, 9], [1, 2], [3, 4]),
        ([0, 1, 6, 7, 8, 9], [2, 3], [4, 5]),
        ([0, 1, 2, 7, 8, 9], [3, 4], [5, 6]),
        ([0, 1, 2, 3, 8, 9], [4, 5], [6, 7]),
        ([0, 1, 2, 3, 4, 9], [5, 6], [7, 8]),
        ([0, 1, 2, 3, 4, 5], [6, 7], [8, 9]),
        ([1, 2, 3, 4, 5, 6], [7, 8], [9, 0]),
        ([2, 3, 4, 5, 6, 7], [8, 9], [0, 1]),
        ([3, 4, 5, 6, 7, 8], [9, 0], [1, 2]),
    ]

Conclusion
----------

The ``Controller`` class provides a robust framework for managing how the data is assigned to different sets.