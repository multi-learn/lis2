Trackers
========

This module provides a framework for tracking and logging metrics during model training and evaluation using various tracking systems, including Weights and Biases (Wandb), MLflow, and CSV logging.

.. currentmodule:: trackers

BaseTracker
-----------------

.. autoclass:: src.trackers.BaseTracker
   :members:
   :undoc-members:
   :show-inheritance:

The ``BaseTracker`` class is an abstract base class that provides a common interface for logging metrics. To create a new tracker, subclass ``BaseTracker`` and implement the ``init``, ``log``, and optionally ``close`` methods.

**Attributes:**
    - **output_run (str)**: Path to the directory where logs are stored.

**Example:**

.. code-block:: python

    class MyCustomTracker(BaseTracker):
        def init(self):
            # Initialize custom tracker
            pass

        def log(self, epoch, log_dict):
            # Log metrics using custom logic
            pass

        def close(self):
            # Clean up resources if necessary
            pass

    tracker = MyCustomTracker(output_run='./logs')
    tracker.init()
    tracker.log(epoch=1, log_dict={'loss': 0.1, 'accuracy': 0.9})
    tracker.close()

Wandb Tracker
-------------

.. autoclass:: src.trackers.Wandb
   :members:
   :undoc-members:
   :show-inheritance:

The ``Wandb`` class is a tracker that uses Weights and Biases (Wandb) for logging metrics. It initializes a Wandb run and logs metrics using the Wandb API.

**Attributes:**
    - **config_schema (dict)**: Configuration schema for validating Wandb parameters.

MLflow Tracker
--------------

.. autoclass:: src.trackers.Mlflow
   :members:
   :undoc-members:
   :show-inheritance:

The ``Mlflow`` class is a tracker that uses MLflow for logging metrics. It initializes an MLflow run and logs metrics using the MLflow API.

**Attributes:**
    - **tracking_uri (str)**: URI for the MLflow server.
    - **experiment_name (str)**: Name of the MLflow experiment.

CSV Logger
----------

.. autoclass:: src.trackers.CsvLogger
   :members:
   :undoc-members:
   :show-inheritance:

The ``CsvLogger`` class is a tracker that logs metrics to a CSV file. It initializes the CSV file with headers and logs metrics as rows in the file.

**Attributes:**
    - **csv_filename (str)**: Path to the CSV file where logs are saved.
    - **file_initialized (bool)**: Indicates whether the CSV file is initialized.
    - **fieldnames (List[str])**: List of field names for the CSV file.

Trackers Manager
----------------

.. autoclass:: src.trackers.Trackers
   :members:
   :undoc-members:
   :show-inheritance:

The ``Trackers`` class manages multiple trackers and coordinates logging. It initializes all trackers based on the provided configurations and logs metrics to all initialized trackers.

**Attributes:**
    - **loggers (List[BaseTracker])**: List of initialized tracker instances.
    - **loggers_configs (List[Dict[str, any]])**: Configuration for the trackers.
    - **output_run (str)**: Path to the directory where logs are stored.
    - **is_init (bool)**: Indicates whether trackers are initialized.

**Example:**

.. code-block:: python

    trackers_manager = Trackers(loggers_configs=[
        {"type": "Wandb", "entity": "my_entity"},
        {"type": "Mlflow", "tracking_uri": "http://localhost:5000", "experiment_name": "my_experiment"}
    ], output_run='./logs')

    trackers_manager.init()
    trackers_manager.log(epoch=1, log_dict={'loss': 0.1, 'accuracy': 0.9})
    trackers_manager.finish()

Conclusion
----------

This module provides a flexible and extensible framework for tracking and logging metrics during model training and evaluation. By leveraging the ``BaseTracker`` class and its subclasses, you can easily integrate and customize trackers for your specific needs.
