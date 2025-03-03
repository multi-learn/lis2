Data Augmentation
=================
This module provides a framework for integrating ``Torchvision Transforms`` with a configurable schema, allowing dynamic configuration and extension of transforms.
One can also add its Custom Data Augmentation by inheriting from ``BaseDataAugmentation``.

For more information about Torchvision Transforms : `Torchvision Transforms <https://pytorch.org/vision/0.9/transforms.html>`_

Usage
-----

This module is designed to be used inside a :ref:`Datasets`, thus the configuration relative to the data augmentation is a part of the configuration
of the dataset. Here is an example configuration :

.. code-block:: python

    data_augmentations= [
        {"type": "ToTensor"},
        {"type": "CenterCrop", "name": "center_crop", "size": 10},
        {"type": "NoiseDataAugmentation", "name": "input", "to_augment": [0, 1], "keys_to_augment":["patch"]}
    ]

Let's dissect the components to have a better understand:

- Each of the listed augmentation will be applied sequentially, one after the other, in the specified order,
- ``ToTensor`` must be the **first** augmentation and applied to all dimensions, **except** if ``ExtendedDataAugmentation`` is used. In this case, it must be second, behind the latter,
- The parameters for transforms from ``torchvision`` take the same parameters as in the ``torchvision`` documentation,
- The ``name`` parameter is mendatory, and useful to differenciate different augmentations that use the same ``Augmentation class``,
- By default, each ``augmentation`` is performed on **every** dimension of the data,
- If you want to apply an augmentation only to a specific part of the data, you can specify it with ``keys_to_augment``,
- Make sure that ``keys_to_augment`` correspond to the name of the variables in the ``__getitem__`` of your ``dataset``,

DataAugmentations
-----------------

.. automodule:: src.datasets.data_augmentation.DataAugmentations
   :members:
   :undoc-members:
   :show-inheritance:

Augmentation Zoo
----------------

NoiseDataAugmentation
*********************

.. automodule:: src.datasets.data_augmentation.NoiseDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

ExtendedDataAugmentation
************************

.. automodule:: src.datasets.data_augmentation.ExtendedDataAugmentation
   :members:
   :undoc-members:
   :show-inheritance:

Custom Augmentations and Torchvion Transforms
---------------------------------------------

Now let’s say we want to create a class to use the data_augmentations from torchvision.transforms as well as custom transforms.

We have our base class:

.. code-block:: python

    class BaseDataAugmentation(abc.ABC, TypedConfigurable):

        @abc.abstractmethod
        def __call__(self):
            pass

Then, let’s add all the transforms from `torchvision.transforms` to this class

.. code-block:: python

    import torchvision.transforms.transforms as transforms

    def register_transforms() -> None:
        """
        Registers all valid transforms classes from torch.optim as subclasses of BaseDataAugmentation.
        """
        EXCLUDE_TRANSFORMS = ["compose"]
        transforms_classes = inspect.getmembers(transforms, inspect.isclass)
        for name, cls in transforms_classes:
            if (
                name == "Compose"
                or isinstance(cls, Enum)
                or issubclass(cls, Enum)
                or issubclass(cls, torch.Tensor)
            ):
                continue
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

using these functions : 

.. code-block:: python

    def generate_config_schema(transform_class) -> Dict[str, Schema]:
        """
        Automatically generates a configuration schema for a given transform
        by inspecting its __init__ parameters.

        Args:
            transform_class (Type[Optimizer]): The optimizer class to generate the schema for.

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
                param_type = Any  # Default to string if no annotation or default

            config_schema[param_name] = Schema(
                type=param_type,
                optional=optional,
                default=default,
            )
        return config_schema


and 

.. code-block:: python

    def infer_type_from_default(default_value: Any) -> Any:
        """
        Infers the type annotation from the default value.

        Args:
            default_value (Any): The default value to infer the type from.

        Returns:
            Any: The inferred type.
        """
        if isinstance(default_value, tuple):
            # Infer types of elements in the tuple
            element_types = tuple(type(element) for element in default_value)
            # Create a typing.Tuple with the inferred element types
            return Tuple[element_types]
        elif isinstance(default_value, list):
            # Infer the type of elements in the list
            element_type = type(default_value[0]) if default_value else Any
            return List[element_type]
        else:
            return type(default_value)


Now, we added (nearly) all the `transforms` methods from `torchvision.transforms` to our class and we can call them using our typical schema :

.. code-block:: python

    config = {"type": "CenterCrop",
            "name": "center_crop",
            "size": "10"}

    BaseDataAugmentation.from_config(config)

Custom Data Augmentation Class 
------------------------------

Now let’s create a custom data augmentation class :

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
            Apply a data augmentation scheme to given data
                    """
                    
                    # Augmentation code 
                    return new_data


Apply data augmentations sequentially
-------------------------------------

We want to use them sequentially, so we provide a class to do that :

.. code-block:: python

    class DataAugmentations:
        def __init__(self, augmentations_configs: List[Dict[str, Any]]):
            """
            Initializes the Augmentations class with a list of augmentation configurations.

            Args:
                augmentations_configs (List[Dict[str, Any]]): List of augmentations configurations.
            """
            self.augmentations = [
                BaseDataAugmentation.from_config(config) for config in augmentations_configs
            ]

        def compute(self, data):
            for augmentation in self.augmentations:
                data = augmentation(data)
            return data


we obtain this final code :

.. code-block:: python

    data_augmentations= [
        {"type": "ToTensor"},
        {"type": "CenterCrop",
        "name": "center_crop",
        "size": 10},
        {"type": "NoiseDataAugmentation",
        "name": "input",
        "to_augment": [0, 1],
        "keys_to_augment":["patch"]}
    ]

.. note::
    The order of the augments matters. ``ToTensor`` Must be the first augmentation, except in one case: if ``ExtendedDataAugmentation`` is provided. Indeed, for performances reasons,
    we need to use ``numpy`` instead of ``torch`` for this augment.


We can use it this way :

.. code-block:: python

    data_augment = DataAugmentations(augmentations_configs=data_augmentations)