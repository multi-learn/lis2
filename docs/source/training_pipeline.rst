Pipeline
========

This module provides the `KfoldsTrainingPipeline` class, which is responsible for managing the k-fold pipeline.

KfoldsTrainingPipeline Class
----------------------------

The ``KfoldsTrainingPipeline`` class is responsible for managing the interractions between :ref:`controller`, :ref:`datasets`, :ref:`trainer` in the context of a `k-fold` training.

.. note::

    You can train on multiple GPUs by setting the `gpus` parameter to the list of GPU ids you want to use. By default, the pipeline will use all available GPUs.

.. autoclass:: src.pipeline.KfoldsTrainingPipeline
   :members:
   :undoc-members:
   :show-inheritance:

K-fold Training Pipeline
------------------------

The `run_training` method orchestrates the process, including different phases Here's a detailed explanation of its internal workings:

1. **K-folds handling:**

   - Calls the :ref:`controller` which contains the splits as well as the assignment of each area to a fold.

2. **Looping on each split:**

   - For each split (i.e. each configuration of "which fold is in which set"), train, valid and test sets are created using the indices in fold assignments
   - A :ref:`trainer` is loaded 
   - A :ref:`models` is trained using this trainer.
   - This loop is repeated k-times. See more details on the training in :ref:`trainer`

3. **Results and metrics:**
   - Results and metrics are logged for each split

4. **Inference:**

   - Inference can be run using each trained model on each test set, in order to have an inference on the whole image.

Conclusion
----------

The `Pipeline` class orchestrates the different block that can be involved in the training of a model in the context of k-folds.
