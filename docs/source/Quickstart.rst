Quickstart
==========


Installation
------------

To install this package, we recommend that you use Conda to create a virtual environment and install the dependencies:

.. code-block:: bash

    conda env create -f environment.yml

Test Your Installation
----------------------

Once the installation is done, you should be able to run the tests:

.. code-block:: bash

    pytest tests

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

    python scripts/main_build_mosaic.py -c ./configs/config_mosaic.yaml

An example of configuration is provided in configs/config_preprocess.yaml. Make sure to adapt it to your folder organization. Then run the following command:

.. code-block:: bash

    python scripts/main_preprocessing.py -c ./configs/config_preprocess.yaml

At the end of the preprocessing, you should have all the unified ``fits`` files in ``output_folder`` and a ``./h5`` file containing all the patches in possibility another ``output_folder``.

2. Training pipeline

When you want to train a model, two possibilities arise : you either want to use the ``standard`` training process or the ``k-folds`` training process. For the first, ``main_train`` is provided. For the latter, ``main_training_k_fold`` is provided. 
Both have their respective configurations in ``./configs/``. Make sure to adapt them.

For more details about the ``standard`` training, please refer to :ref:`trainer`.

.. code-block:: bash

    python scripts/main_train.py -c ./configs/main_train.yaml


For more details about the ``k-folds`` training, please refer to :ref:`pipeline`

.. code-block:: bash

    python scripts/main_train.py -c ./configs/main_training_k_fold.yaml


3. Inference
Run segmentation. TODO
