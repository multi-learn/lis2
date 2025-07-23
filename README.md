
![Galactic filaments](./docs/source/_static/lis2-normal.png)

#  BigSF

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

3. Run segmentation. # TODO



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

#TODO

</details>

## Dev guide

<details>
<summary>For details about how to use the library for your own use case</summary>

Le package BigSF permet d'ajouter ou de modifier tout composant de manière modulaire grâce à l'architecture basée sur
Configurable et les schémas de configuration. Tous les composants (modèles, datasets, optimisateurs, métriques, etc.)
suivent ce principe.

---

### Architecture Modulaire avec Configurable et TypedConfigurable

**BigSF** repose sur une architecture modulaire grâce aux classes de base `Configurable` et `TypedConfigurable`. Ces
classes permettent une configuration flexible, extensible et standardisée des composants (modèles, datasets,
optimiseurs, etc.).

#### 1. **Configurable** : Création Dynamique de Composants

`Configurable` est une classe de base qui utilise des schémas (`Schema`) pour valider dynamiquement les configurations.
Elle permet:

- **Validation** : Chaque paramètre est validé par type et contrainte avant l’instanciation grace a la classe `Schema`.
- **Flexibilité** : Chargement des configurations depuis des dictionnaires Python ou des fichiers YAML. Les
  configuration sont dynamique car les paramètres dependent du type d'objet/classe demandé .

**Exemple** :

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

`TypedConfigurable` étend `Configurable` en ajoutant la possibilité de choisir dynamiquement une sous-classe à
instancier en fonction d’un paramètre `type`.

**Exemple avec Modèles** :

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

### Fonctionnement des Schémas

#### Concept de la Classe `Schema`

Schema définit la structure attendue pour chaque paramètre de configuration. Elle joue un rôle central dans la
validation et l'application des valeurs par défaut lors de l'instanciation des objets.

Attributs principaux de Schema :

- `type` : Spécifie le type attendu (e.g., int, float, str).
- `default` : Définit une valeur par défaut si le paramètre n'est pas fourni.
- `optional` : Indique si le paramètre est optionnel.
- `aliases` : Permet d'utiliser des noms alternatifs pour un même paramètre.

### Ajouter un Composant en Pratique

#### `Configurable`

Définir la classe : Héritez de la classe de base appropriée (ex. BaseModel, Metric, BaseDataset, etc.) ou directement de
```Configurable``` et implémentez la logique nécessaire.

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

Ajouter dans la configuration YAML : Référencez le nouveau composant avec ses paramètres.

```yaml
component:
  param1: "example"
```

Utiliser dans le pipeline : Chargez et intégrez dynamiquement le composant via from_config.

```python
import NewComponent

component = NewComponent.from_config(config['component'])
```

#### `TypedConfigurable`

Définir les sous-classes : Créez des sous-classes pour chaque type de composant.

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

Il est possible d'ajouter des composants supplémentaires en suivant le même processus, tant qu'ils héritent de
``` Configurable```.
De plus, pour certaine classe, comme les modèles, des classe de base sont déjà définies pour faciliter l'ajout de
nouveaux composants:

- ```BaseModel```: Classe de base pour les modèles (` models/base_model.py`).
- ```BaseDataset```: Classe de base pour les datasets (` datasets/dataset.py`).
- ```BaseOptimizer```: Classe de base pour les optimiseurs (` core/optim.py`).
- ```BaseScheduler```: Classe de base pour les schedulers (` core/scheduler.py`).
- ```Metric```: Classe de base pour les métriques (` core/metrics.py`).
- ```Encoder```: Classe de base pour les encodeurs (` models/encoders/encoder.py`).
- ```EarlyStopping```: Classe de base pour les early stopping (` core/early_stopping.py`).

---

## Tests et Validation

### Tests Unitaires

Les tests unitaires sont inclus dans le package pour garantir le bon fonctionnement des composants. Pour exécuter les
tests, utilisez le dossier `tests`.

### Trainer

Le package BigSF fournit un **Trainer** prêt à l'emploi pour orchestrer l'entraînement, la validation, et le test des
modèles. Le **Trainer** s'intègre dynamiquement à l'ensemble des composants configurés dans un fichier YAML.

#### Exemple de Configuration YAML

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

- **Résultats** : Les métriques et pertes sont enregistrées dans le répertoire défini par `output_dir`.
- **Snapshots** : Les fichiers de snapshot contiennent :
  - L’état du modèle (`MODEL_STATE`).
  - L’état des optimiseurs et des schedulers.
  - La configuration globale (`GLOBAL_CONFIG`).

#### Reprise à Partir d’un Snapshot

Pour reprendre un entraînement depuis un snapshot :

```python
from core.trainer import Trainer

trainer = Trainer.from_snapshot("./results/experiment_1/best.pt")
trainer.train() 
```

---
</details>
