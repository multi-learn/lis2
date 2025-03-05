Data Augmentation
=================

This module provides a framework for integrating **Torchvision Transforms** with a configurable schema, allowing dynamic configuration and extension of transforms. It also enables the addition of custom data augmentations by inheriting from the ``BaseDataAugmentation`` class.

For more information about Torchvision Transforms, please refer to the `Torchvision Transforms documentation <https://pytorch.org/vision/stable/transforms.html>`_.

Overview
--------

The module is designed to be used within a dataset, where the data augmentation configuration becomes part of the dataset configuration. Each augmentation is applied sequentially, following the order specified in the configuration list.

.. note::
    **Important: Management of ``ToTensor``**
        - The configuration should include a ``ToTensor`` augmentation to ensure that the final output is a Tensor.
        - If no ``ToTensor`` augmentation is provided by the user, the module automatically **appends a default ``ToTensor`` transform** to the configuration.
        - Although there is no strict constraint regarding the order of augmentations, care must be taken when adding new augmentations. The placement of the ``ToTensor`` transform should be handled similarly to a Torch Compose (see: `torchvision.transforms.Compose <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose>`_) to avoid type errors.
        - **Note:** The input to the augmentation pipeline is expected to be NumPy arrays, while the final output must be a ``torch.Tensor``.

Usage
-----

This module is intended to be used as part of a dataset. Below is an example configuration:

.. code-block:: python

    data_augmentations = [
        {"type": "ToTensor"},
        {"type": "CenterCrop", "name": "center_crop", "size": 10},
        {"type": "NoiseDataAugmentation", "name": "input", "to_augment": [0, 1], "keys_to_augment": ["patch"]}
    ]

Key points:

- **Sequential Application:** Each augmentation is applied one after the other in the specified order.
- **Mandatory ``name`` Parameter:** The ``name`` parameter is required to differentiate between different augmentations even if they use the same augmentation class.
- **Selective Augmentation:** Use the parameter ``keys_to_augment`` to specify the particular data elements to augment. These keys must correspond to the variables returned by the dataset's ``__getitem__`` method.

DataAugmentations Class
-----------------------

The main class responsible for applying the augmentations is ``DataAugmentations``. Its main functionalities include:

- **Configuration Validation:** Ensures the provided configuration is a list.
- **Default Addition of ``ToTensor``:**
  If the configuration does not include a ``ToTensor`` augmentation, one is automatically appended.
- **Sequential Processing:** Iterates through the list of configured augmentations and applies each to the input data. If an augmentation specifies ``keys_to_augment``, it will be applied only to those keys.
- **Final Type Check:** After applying all augmentations, a type check is performed to ensure that each augmented data element is a ``torch.Tensor`` (or a list of ``torch.Tensor`` objects). If not, an error is raised indicating a potential misplacement of the ``ToTensor`` transform.

Example:

.. code-block:: python

    class DataAugmentations:
        def __init__(self, augmentations_configs: List[Dict[str, Any]]):
            """
            Initializes the Augmentations class with a list of augmentation configurations.

            Args:
                augmentations_configs (List[Dict[str, Any]]): List of augmentation configurations.
            """
            if not isinstance(augmentations_configs, list):
                raise ValueError("augmentations_configs must be a list")

            if not augmentations_configs:
                warnings.warn("Empty list detected. Adding default ToTensor augmentation.", UserWarning)
                has_to_tensor = False
            else:
                has_to_tensor = any(config.get("type") == "ToTensor" for config in augmentations_configs)
            self.augmentations_configs = augmentations_configs
            if not has_to_tensor:
                augmentations_configs.append({"type": "ToTensor"})
            self.augmentations = [BaseDataAugmentation.from_config(config) for config in augmentations_configs]

        def compute(self, data):
            for augmentation in self.augmentations:
                to_augment = getattr(augmentation, "keys_to_augment", [])
                if to_augment:
                    for k, v in data.items():
                        if k in to_augment:
                            try:
                                data[k] = augmentation(v)
                            except (AttributeError, TypeError) as e:
                                raise RuntimeError(
                                    f"Error applying augmentation '{augmentation.__class__.__name__}' to key '{k}' (value type: {type(v)}). "
                                    "Check the placement of ToTensor to ensure proper type conversion."
                                ) from e
                        else:
                            data[k] = v
                else:
                    for k, v in data.items():
                        try:
                            data[k] = augmentation(v)
                        except (AttributeError, TypeError) as e:
                            raise RuntimeError(
                                f"Error applying augmentation '{augmentation.__class__.__name__}' to key '{k}' (value type: {type(v)}). "
                                "Check the placement of ToTensor to ensure proper type conversion."
                            ) from e

            # Final type check: The output must be torch.Tensor(s)
            assert all([isinstance(v, (torch.Tensor, list)) for _, v in data.items()]), (
                f"End of augmentation pipeline must return Tensor(s). Got {[type(v) for v in data.values()]}. "
                'Ensure that a proper ToTensor augmentation is included at the correct position in the pipeline.'
            )

            return list(data.values())

Augmentation Zoo
----------------

The module provides several built-in augmentation classes, including:

NoiseDataAugmentation
*********************

A custom augmentation for adding Gaussian noise to data. It applies noise to specific keys if provided and clamps the output between 0 and 1.

.. automodule:: src.datasets.data_augmentation.NoiseDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

ExtendedDataAugmentation
************************

An advanced augmentation class that applies multiple transformations using NumPy for performance reasons. When using this augmentation, it is crucial to ensure that the pipeline includes a ``ToTensor`` transform at the appropriate position to convert NumPy arrays back to Tensors.

.. automodule:: src.datasets.data_augmentation.ExtendedDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

Custom Augmentations and Torchvision Transforms
-------------------------------------------------

You can extend the augmentation framework to include both Torchvision transforms and custom augmentations.

Base Class for Augmentations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class BaseDataAugmentation(abc.ABC, TypedConfigurable):
        @abc.abstractmethod
        def __call__(self, data):
            pass

Registering Torchvision Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All valid transform classes from ``torchvision.transforms`` are automatically registered as subclasses of ``BaseDataAugmentation`` using the following function:

.. code-block:: python

    import torchvision.transforms.transforms as transforms
    import inspect
    from enum import Enum

    def register_transforms() -> None:
        """
        Registers all valid transforms classes from torchvision.transforms as subclasses of BaseDataAugmentation.
        """
        EXCLUDE_TRANSFORMS = ["compose", "Compose", "Lambda"]
        transforms_classes = inspect.getmembers(transforms, inspect.isclass)
        for name, cls in transforms_classes:
            if (
                name in EXCLUDE_TRANSFORMS
                or isinstance(cls, Enum)
                or issubclass(cls, Enum)
                or issubclass(cls, torch.Tensor)
            ):
                continue
            else:
                if name == "ToTensor":
                    subclass = type(
                        name,
                        (BaseDataAugmentation, cls),
                        {
                            "__module__": __name__,
                            "aliases": [name.lower()],
                            "config_schema": generate_config_schema(cls),
                        },
                    )
                else:
                    subclass = type(
                        name,
                        (BaseDataAugmentation, cls),
                        {
                            "__module__": __name__,
                            "aliases": [name.lower()],
                            "config_schema": generate_config_schema(cls),
                        },
                    )
            globals()[name] = subclass

Generating Configuration Schemas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following helper functions are used to generate a configuration schema automatically by inspecting the ``__init__`` parameters of a transform class:

.. code-block:: python

    def generate_config_schema(transform_class) -> Dict[str, Schema]:
        """
        Automatically generates a configuration schema for a given transform by inspecting its __init__ parameters.

        Args:
            transform_class (Type): The transform class to generate the schema for.

        Returns:
            Dict[str, Schema]: A dictionary mapping parameter names to their corresponding schema.
        """
        config_schema = {}
        init_signature = inspect.signature(transform_class.__init__)
        for param_name, param in init_signature.parameters.items():
            if param_name in ["self", "params", "defaults"]:
                continue

            # Determine if the parameter is optional
            optional = param.default != inspect.Parameter.empty
            default = param.default if optional else None

            # Infer the type
            if param.annotation != inspect.Parameter.empty:
                param_type = param.annotation
            elif param.default != inspect.Parameter.empty:
                param_type = infer_type_from_default(param.default)
            else:
                param_type = Any

            config_schema[param_name] = Schema(
                type=param_type,
                optional=optional,
                default=default,
            )
        return config_schema


    def infer_type_from_default(default_value: Any) -> Any:
        """
        Infers the type annotation from the default value.

        Args:
            default_value (Any): The default value to infer the type from.

        Returns:
            Any: The inferred type.
        """
        if isinstance(default_value, tuple):
            element_types = tuple(type(element) for element in default_value)
            return Tuple[element_types]
        elif isinstance(default_value, list):
            element_type = type(default_value[0]) if default_value else Any
            return List[element_type]
        else:
            return type(default_value)

Using the Configuration Schema
------------------------------

After registration, you can create transforms using a typical configuration schema. For example:

.. code-block:: python

    config = {
        "type": "CenterCrop",
        "name": "center_crop",
        "size": 10
    }

    augmentation = BaseDataAugmentation.from_config(config)

Custom Data Augmentation Class
------------------------------

You can also implement your own custom augmentation class by extending ``BaseDataAugmentation``. For example:

.. code-block:: python

    class NoiseDataAugmentation(BaseDataAugmentation):
        config_schema = {
            "type": Schema(str),
            "input_noise_var": Schema(float, default=0),
            "output_noise_var": Schema(float, default=0),
        }

        def __call__(self, data):
            return self.apply_data_augmentation(data)

        def apply_data_augmentation(self, data):
            """
            Apply a data augmentation scheme to the given data.
            """
            # Custom augmentation code goes here.
            return new_data

Final Assembly of Augmentations
--------------------------------

Finally, the augmentations are applied sequentially by the ``DataAugmentations`` class. A complete configuration might look like this:

.. code-block:: python

    data_augmentations = [
        {"type": "ToTensor"},
        {"type": "CenterCrop", "name": "center_crop", "size": 10},
        {"type": "NoiseDataAugmentation", "name": "input", "to_augment": [0, 1], "keys_to_augment": ["patch"]}
    ]

.. note::
    Although the order of the augmentations is flexible, proper placement of the ``ToTensor`` transform is essential to ensure type consistency. The pipeline expects NumPy arrays as input and must output a ``torch.Tensor``. When adding new augmentations, position the ``ToTensor`` transform similarly to how you would in a Torch Compose (see: `torchvision.transforms.Compose <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Compose>`_) to prevent type errors.
