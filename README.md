
![Galactic filaments](./docs/source/_static/lis2-normal.png)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Release](https://img.shields.io/github/v/release/multi-learn/lis2)](https://github.com/multi-learn/lis2/releases/)
#  LIS² (Large Image Split Segmentation): 
A toolbox for large-scale, single-image semantic segmentation.

## Installation

To install this package, we recommend that you use **Conda** to create a virtual env and install the dependancies :

```bash
conda env create -f environment.yml
```

## Standard Usage

0. Once the installation is done, you should be able to run the test.

```
pytest tests
```

1. Data preprocessing. An example of configuration is proposed in `configs/config_preprocess.yaml`. Make sure to adapt it to your folder organization. Then run the following command.

``` 
python scripts/main_preprocessing.py -c /path/to/your_config.yaml
```

2. Training pipeline. Two possibilities : you can use `scripts/main_train.py` for a standard training, and `scripts/main_training_k_fold.py` for a k-fold training. Make sure to use the example configuration `configs/training.yaml` or `config_kfolds.yaml` according to your use case, with the right paths. 

```
python scripts/main_train.py -c .path/to/your_config.yaml
```

3. Run segmentation.



## Advanced usage

<details>
<summary>For details about the different steps</summary>

### 1. Data preprocessing

In this phase, the data representing the galactic plan, contained in the `.fits` file is transformed into an `h5` file of patches. Each patch is of a defined `patch_size`. If a patch is tempty, it is discarded.

### 2. Training pipeline

The training pipeline is divided into two different use cases : `standard` and `k-folds`. The `standard` use case is a specific case of `k-fold` with `k=1`, so we won't give much details about it as it can be deducted from the `k-folds`.

Now, we develop each of the steps of the training pipeline :

1- Initialization. 

During the init phase, the `fold_controller` is initialized, according to its configuration given in the configuration file. This controller loads the `patches.h5` file, created during the previous phase. Then it generates the k-folds splits according to the configuration, i.e. if `k=4` and `k_train=2`:
``` 
splits = [[[1, 2], [3], [4]], [[3, 4], [1], [2]]]
``` 
with two folds into the train set, and the rest separated into valid and test.

Then, the `controller` assigns each patch to an area, and this area to a fold according to different strategies. We do this `area trick` to ensure the continuity of the data because of some normalization necessities. The default strategy is called `random`, which corresponds to a `round robin` strategy. A naïve strategy can also be used, where the image is divided into `k` equal parts, and then each part is considered as a fold.
The indices of how the area are distributed into the folds are stored, so they don't have to be computed at each run.

2- Training

For each split, the train, valid and test sets are loaded according to the fold assigments. The model to train is loaded according to the configuration, and then k different versions of this model are trained. Some results are saved in log files.


### 3. Run segmentation



</details>

## Dev guide

<details>
<summary>For details about how to use the library for your own use case</summary>

Le package BigSF permet d'ajouter ou de modifier tout composant de manière modulaire grâce à l'architecture basée sur
Configurable et les schémas de configuration. Tous les composants (modèles, datasets, optimisateurs, métriques, etc.)
suivent ce principe.

---

### Modular Architecture with Configurable and TypedConfigurable

**LIS2** is based on a modular architecture thanks to the `Configurable` and `TypedConfigurable` base classes. 
These classes allow for flexible, extensible, and standardized configuration of components (models, datasets, optimizers, etc.).

#### 1. **Configurable** : Dynamic Creation of ComponentsCréation Dynamique de Composants

`Configurable` is a base class that uses schemas to dynamically validate configurations.
It allows:


- **Validation**: Each parameter is validated by type and constraint before instantiation using the `Schema` class.
- **Flexibility**: Load configurations from Python dictionaries or YAML files.
  Configurations are dynamic because the parameters depend on the type of object/class requested.
- 
**Example** :

```python
from configurable import Configurable, Schema


class MyComponent(Configurable):
  config_schema = {
    'learning_rate': Schema(float, default=0.01),
    'batch_size': Schema(int, default=32),
  }

  def __init__(self):


config = {'learning_rate': 0.001, 'batch_size': 64}
component = MyComponent.from_config(config)
print(component.learning_rate)  # 0.001
print(component.batch_size)  # 64
```

#### 2. **TypedConfigurable** : Gestion Dynamique de Sous-Classes

`TypedConfigurable` extends `Configurable` by adding the ability to dynamically choose a subclass to instantiate based on a `type` parameter.

**Example with Models** : 

```python
from configurable import TypedConfigurable, Schema


class BaseModel(TypedConfigurable):
  aliases = ['base_model']


class CNNModel(BaseModel):
  aliases = ['cnn']
  config_schema = {
    'filters': Schema(int, default=32),
    'kernel_size': Schema(int, default=3),
  }

  def __init__(self):


config = {'type': 'cnn', 'filters': 64, 'kernel_size': 5}
model = BaseModel.from_config(config)
print(model.filters)  # 64
print(model.kernel_size)  # 5
```

### How Schemas Work

#### Concept of the  Class `Schema`

Schema defines the expected structure for each configuration parameter. It plays a central role in validating and applying default values ​​when instantiating objects.

Main Schema Attributes:

- `type`: Specifies the expected type (e.g., int, float, str).
- `default`: Defines a default value if the parameter is not provided.
- `optional`: Indicates whether the parameter is optional.
- `aliases`: Allows you to use alternative names for the same parameter.


Schema defines the expected structure for each configuration parameter. 
It plays a central role in validating and applying default values when instantiating objects.


### Adding a Component in Practice

#### `Configurable`

Define the class: Inherit from the appropriate base class (e.g., BaseModel, Metric, BaseDataset, etc.) or directly from
```Configurable``` and implement the necessary logic.


```python
class NewComponent(Configurable):
  config_schema = {
    'param1': Schema(str),
    'param2': Schema(int, default=10),
  }

  def __init__(self):
    print(self.param1)
    print(self.param2)
```
Add in YAML configuration: Reference the new component with its parameters.

```yaml
component:
  param1: "example"
```

Use in pipeline: Dynamically load and integrate the component via from_config.

```python
import NewComponent

component = NewComponent.from_config(config['component'])
```

#### `TypedConfigurable`

Define subclasses: Create subclasses for each component type.

```python
class NewComponent(Configurable):
  config_schema = {
    'param1': Schema(str, default="default"),
    'param2': Schema(int, default=10),
  }

  def __init__(self):
    print(self.param1)  # default
    print(self.param2)  # 10


class NewComponentNeg(Configurable):
  config_schema = {
    'param1': Schema(str, default="Neg"),
    'param2': Schema(int, default=-10),
  }

  def __init__(self):
    print(self.param1)  # Neg
    print(self.param2)  # -10
```

```yaml
component:
  type: NewComponentNeg
  param1: "example"
```

```python
import Configurable

component = Configurable.from_config(config['component'])

# Output
# example
# -10
```
Additional components can be added using the same process, as long as they inherit from ``` Configurable```.
In addition, for some classes, such as models, base classes are already defined to facilitate the addition of new components:



- ```BaseModel```: Base class for models (` models/base_model.py`).
- ```BaseDataset```: Base class for datasets (` datasets/dataset.py`).
- ```BaseOptimizer```: Base class for optimizers (` core/optim.py`).
- ```BaseScheduler```: Base class for schedulers (` core/scheduler.py`).
- ```Metric```: Base class for metrics (` core/metrics.py`).
- ```Encoder```: Base class for encoders (` models/encoders/encoder.py`).
- ```EarlyStopping```: Base class for early stopping (` core/early_stopping.py`).

---

## Tests and Validation

### Unit Tests

Unit tests are included in theClasse de base pour les modèles (` models/base_model.py`).

- ```BaseDataset```: Base class for datasets (` datasets/dataset.py`).
- ```BaseOptimizer```: Base class for optimizers (` core/optim.py`).
- ```BaseScheduler```: Base class for schedulers (` core/scheduler.py`).
- ```Metric```: Base class for metrics (` core/metrics.py`).
- ```Encoder```: Base class for encoders (` models/encoders/encoder.py`).
- ```EarlyStopping```: Base class for early stopping (` core/early_stopping.py`).

---

## Tests and Validation

### Unit Tests

Unit tests are included in the` PyTorch` class (` datasets/dataset.py`).
- ```BaseOptimizer```: Class for optimizers (` core/optim.py`).
- ```BaseScheduler```: Class for schedulers (` core/scheduler.py`).
- ```Metric```: Class for metrics (` core/metrics.py`).
- ```Encoder```: Class for encoders (` models/encoders/encoder.py`).
- ```EarlyStopping```: Class for early stopping (` core/early_stopping.py`).

---

## Tests and Validation

### Units Tests

Unit tests are included in the package to ensure the components function properly. To run the tests, use the `tests` folder.

### Trainer

The **LIS²** package provides a ready-to-use **Trainer** to orchestrate model training, validation, and testing.
The **Trainer** dynamically integrates with all the components configured in a YAML file.

#### Example of Configuration YAML file

```yaml
trainer:
  output_dir: ./results
  run_name: experiment_1
  model:
    type: unet
    in_channels: 1
    out_channels: 1
  optimizer:
    type: adam
    lr: 0.001
  epochs: 50
  batch_size: 16
  metrics:
    - type: dice
```

#### Outputs


- **Results**: Metrics and losses are saved in the directory defined by `output_dir`.
- **Snapshots**: Snapshot files contain:
- The model state (`MODEL_STATE`).
- The state of optimizers and schedulers.
- The global configuration (`GLOBAL_CONFIG`).
- 
#### Resuming from a Snapshot

Resuming from a Snapshot:

```python
from core.trainer import Trainer

trainer = Trainer.from_snapshot("./results/experiment_1/best.pt")
trainer.train() 
```

---
</details>
