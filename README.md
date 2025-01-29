# Package BigSF

## Installation

Pour installer ce package, vous pouvez utiliser **Conda** avec les outils de développement inclus :

```bash
conda env create -f environment.yml
```

## Usage

## Concepts Clefs

Le package BigSF permet d'ajouter ou de modifier tout composant de manière modulaire grâce à l'architecture basée sur
Customizable et les schémas de configuration. Tous les composants (modèles, datasets, optimisateurs, métriques, etc.)
suivent ce principe.

---

### Architecture Modulaire avec Customizable et TypedCustomizable

**BigSF** repose sur une architecture modulaire grâce aux classes de base `Customizable` et `TypedCustomizable`. Ces
classes permettent une configuration flexible, extensible et standardisée des composants (modèles, datasets,
optimiseurs, etc.).

#### 1. **Customizable** : Création Dynamique de Composants

`Customizable` est une classe de base qui utilise des schémas (`Schema`) pour valider dynamiquement les configurations.
Elle permet:

- **Validation** : Chaque paramètre est validé par type et contrainte avant l’instanciation grace a la classe `Schema`.
- **Flexibilité** : Chargement des configurations depuis des dictionnaires Python ou des fichiers YAML. Les
  configuration sont dynamique car les paramètres dependent du type d'objet/classe demandé .

**Exemple** :

```python
from configs.config import Customizable, Schema


class MyComponent(Customizable):
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

#### 2. **TypedCustomizable** : Gestion Dynamique de Sous-Classes

`TypedCustomizable` étend `Customizable` en ajoutant la possibilité de choisir dynamiquement une sous-classe à
instancier en fonction d’un paramètre `type`.

**Exemple avec Modèles** :

```python
from configs.config import TypedCustomizable, Schema


class BaseModel(TypedCustomizable):
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

#### `Customizable`

Définir la classe : Héritez de la classe de base appropriée (ex. BaseModel, Metric, BaseDataset, etc.) ou directement de
```Customizable``` et implémentez la logique nécessaire.

```python
class NewComponent(Customizable):
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

#### `TypedCustomizable`

Définir les sous-classes : Créez des sous-classes pour chaque type de composant.

```python
class NewComponent(Customizable):
  config_schema = {
    'param1': Schema(str, default="default"),
    'param2': Schema(int, default=10),
  }

  def __init__(self):
    print(self.param1)  # default
    print(self.param2)  # 10


class NewComponentNeg(Customizable):
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
import Customizable

component = Customizable.from_config(config['component'])

# Output
# example
# -10
```

Il est possible d'ajouter des composants supplémentaires en suivant le même processus, tant qu'ils héritent de
``` Customizable```.
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