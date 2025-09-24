Trainer
==============

.. currentmodule:: trainer

This module provides the `Trainer` class, which is responsible for managing the training, validation, and testing loops of machine learning models. It supports distributed training, early stopping, and various tracking mechanisms.


Training Process
----------------

The ``train()`` method orchestrates the training process, including validation and testing phases. Here's a detailed explanation of its internal workings:

1. **Initialization:**
   - The method begins by initializing the trackers if the current process is the main GPU.
   - It sets up a progress bar for tracking epochs.

2. **Training Loop:**
   - For each epoch, the method runs the training loop (`_run_loop_train`) to compute the training loss.
   - It then runs the validation loop (`_run_loop_validation`) if a validation dataset is provided.

3. **Logging and Saving:**
   - Metrics, including training and validation loss, are logged using the trackers.
   - The model is saved if the current training loss is the best observed so far.
   - The model is also saved at regular intervals defined by `save_interval`.
   - If `save_last` is enabled, the model is saved at the end of each epoch.

4. **Early Stopping:**
   - The method checks if early stopping criteria are met using the `early_stopper`. If so, training is halted early.

5. **Finalization:**
   - After training, the trackers are finalized, and a summary of the training process is logged.

**Subtleties:**

- **Distributed Training:** The trainer supports distributed training by wrapping the model with `DistributedDataParallel` if multiple GPUs are available.
- **Metrics Tracking:** Metrics are updated and logged during both training and validation loops.
- **Snapshot Saving:** Snapshots of the model are saved with detailed information, including model state, optimizer state, and training information.

.. code-block:: python

    snapshot = {
            "MODEL": {
                "MODEL_CONFIG": self.config["model"],
                "MODEL_STATE": self.model.state_dict(),
            },
            "TRAIN_INFO": {
                "EPOCHS_RUN": epoch,
                "BEST_LOSS": loss,
                "OPTIMIZER_STATE": self.optimizer.state_dict(),
                "SCHEDULER_STATE": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            },
            "GLOBAL_CONFIG": self.global_config.to_dict(),
        }

Trainer Class
-------------

The ``Trainer`` class is responsible for managing the complete lifecycle of model training. It handles configuration setup, data loading, training loops, validation, and testing.

.. autoclass:: lis2.trainer.Trainer
   :members:
   :undoc-members:
   :show-inheritance:
