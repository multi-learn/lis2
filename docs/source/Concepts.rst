Concepts
========

Core Classes
************

The toolbox is centered around two main classes: ``Configurable`` and ``TypedConfigurable`` from `configurable-cl <https://pypi.org/project/configurable-cl/>`_ lib.

Configurable
************

The Configurable class provides dynamic component creation using a defined config_schema. It handles parameter validation, assigns configuration parameters as instance attributes, and performs precondition checks during instantiation. If you use Configurable, you must use from_config(...) to instantiate. You can use __init__ for custom initialization but you lose the automatic validation, automatic adding attributes and preconditionning.

Example:

.. code-block:: python

    from configurable import Configurable, Schema

    class MyComponent(Configurable):
        config_schema = {
            'learning_rate': Schema(float, default=0.01),
            'batch_size': Schema(int, default=32),
        }

        def preconditions(self):
            assert self.learning_rate > 0, "Learning rate must be positive"

        def __init__(self):
            # Custom initialization if needed
            pass

TypedConfigurable
*****************

TypedConfigurable extends Configurable to support dynamic subclass selection based on a type parameter. This approach allows you to define a hierarchy of component implementations and select the appropriate one at runtime.

Example with Abstract Base Classes:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from configurable import TypedConfigurable, Schema
    import abc

    class BaseComponent(TypedConfigurable, abc.ABC):
        aliases = ['base_component']

        @abc.abstractmethod
        def process(self):
            pass

    class SpecificComponentA(BaseComponent):
        aliases = ['component_a']
        config_schema = {
            'param1': Schema(int, default=10),
        }

        def process(self):
            return f"Processing with param1: {self.param1}"

    class SpecificComponentB(BaseComponent):
        aliases = ['component_b']
        config_schema = {
            'param2': Schema(str, default="default_value"),
        }

        def process(self):
            return f"Processing with param2: {self.param2}"

    # Example of dynamic instantiation:
    config_a = {'type': 'component_a', 'param1': 20}
    component_a = BaseComponent.from_config(config_a)
    print(component_a.process())

    config_b = {'type': 'component_b', 'param2': "custom_value"}
    component_b = BaseComponent.from_config(config_b)
    print(component_b.process())

Nested & Hierarchical Configuration
***********************************

One of the library’s key strengths is its support for nested configurations. For example, in an AI pipeline, you might configure a data preprocessor, a model, and an optimizer, each with its own set of parameters:

.. code-block:: yaml

    pipeline:
      data_preprocessor:
        type: 'preprocessor'
        params:
          normalization: true
          resize: 256
      model:
        type: 'advanced_model'
        params:
          layers: 50
          dropout: 0.5
      optimizer:
        type: 'adam_optimizer'
        params:
          learning_rate: 0.001


Each block (e.g., data_preprocessor, model, optimizer) can represent a Configurable or TypedConfigurable component, ensuring a consistent and validated configuration across your system.

