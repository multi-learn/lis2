Quickstart
==========

Installation with pip
---------------------

LIS² can be installed from PyPI using ``pip`` (once the package is published).

We recommend using a virtual environment to avoid conflicts with other Python packages.

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   python -m pip install --upgrade pip
   pip install "lis2[cpu]"

The ``cpu`` extra installs the CPU version of PyTorch and torchvision.
For GPU-enabled installations, see the :ref:`installation options <installation-options>`.


.. _installation-options:

Installation options
--------------------

LIS² provides the following installation options:

.. code-block:: bash

   # CPU installation
   pip install "lis2[cpu]"

   # CUDA 13.2 installation
   pip install "lis2[cu132]"

The CUDA installation requires a compatible NVIDIA GPU and driver.

Development installation
------------------------

To contribute to LIS² or run the test suite, clone the repository:

.. code-block:: bash

   git clone https://github.com/multi-learn/lis2.git
   cd lis2

We recommend using `uv <https://docs.astral.sh/uv/>`_ to manage the development environment:

.. code-block:: bash

   uv sync --extra cpu --group dev

This installs the project, its CPU dependencies and the development tools.

Test your installation
----------------------

Run the test suite with:

.. code-block:: bash

   uv run --no-sync pytest

To generate a coverage report:

.. code-block:: bash

   uv run --no-sync pytest --cov=lis2 --cov-report=term-missing --cov-report=html

The HTML coverage report is generated in ``htmlcov/index.html``.


Standard Usage
--------------

The library is mostly divided into three major steps :
    - ``data preprocessing``,
    - ``training pipeline``,
    - ``inference``.

For each step, a ``configuration`` file is necessary. Example configurations are provided in ``configs`` folder. A ``main`` script is provided for each use case in the ``scripts`` folder.

1. Data preprocessing

The ``data preprocessing`` step is divided into two parts : :ref:`mosaic building` and :ref:`patch extraction`. The first is responsible for creating adapted fits files from the original data, and the latter is responsible for storing the patches
into a HDF5 file, using the patch size provided by the user in the ``config`` file. Before running the scripts, make sure to adapt the paths in the ``config`` file to your folder organization.

.. code-block:: bash

    python scripts/main_build_mosaic.py -c ./configs/config_mosaic.yml

An example of configuration is provided in configs/config_preprocess.yaml. Make sure to adapt it to your folder organization. Then run the following command:

.. code-block:: bash

    python scripts/main_preprocessing.py -c ./configs/config_preprocess.yml

At the end of the preprocessing, you should have all the unified ``fits`` files in ``output_folder`` and a ``./h5`` file containing all the patches in possibility another ``output_folder``.

2. Training pipeline

When you want to train a model, two possibilities arise : you either want to use the ``standard`` training process or the ``k-folds`` training process. For the first, ``main_train`` is provided. For the latter, ``main_training_k_fold`` is provided.
Both have their respective configurations in ``./configs/``. Make sure to adapt them.

For more details about the ``standard`` training, please refer to :ref:`trainer`.

.. code-block:: bash

    python scripts/main_train.py -c ./configs/config_training.yml


For more details about the ``k-folds`` training, please refer to :ref:`pipeline`

.. code-block:: bash

    python scripts/main_training_k_folds.py -c ./configs/config_kfolds.yml


3. Inference
Run segmentation.

.. code-block:: bash

    python scripts/main_segmentation.py -c ./configs/config_infer.yml

Custom Usage
------------

If you want to add and use custom elements (model, dataset ...etc), you can refer to :

- :ref:`adding_model`.
- :ref:`adding_dataset`.
- :ref:`adding_augment`.
- :ref:`adding_controller`.
- :ref:`adding_loss`.
- :ref:`adding_scheduler`.
- :ref:`adding_early_stop`.
- :ref:`metrics`.
- :ref:`trackers`.

You can also apply this tutorial to all modules inherited from ``TypedConfigurable``.
Run segmentation.

