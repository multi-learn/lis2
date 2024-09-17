import inspect
import itertools
import warnings

import yaml


class GlobalConfig:
    """
    Module for storing global configuration data.

    This module provides a global configuration object that can be accessed by all parts of the code. The configuration
    data is stored as attributes of the module.
    """
    _config = {}

    def __setitem__(self, name, value):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        if name not in self._config:
            raise KeyError(f"GlobalConfig does not have key: {name}")
        expected_type = self._config[name]['type']
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Invalid type for key '{name}' in GlobalConfig. Expected {expected_type}, got {type(value)}.")
        self._config[name]['value'] = value

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        if name not in self._config:
            raise KeyError(f"GlobalConfig does not have key: {name}")
        return self._config[name]['value']

    def __str__(self):
        def recursive_str(d, indent=0):
            string = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    string += f"{' ' * indent}{key}:\n{recursive_str(value, indent + 2)}"
                else:
                    string += f"{' ' * indent}{key}: {value}\n"
            return string

        config_string = ""
        config_string += recursive_str(self._config)
        return config_string

    def to_dict(self):
        return {key: value['value'] for key, value in self._config.items()}


class Customizable:
    """
    Base class for Customizable objects.

    This class provides a mechanism for creating objects from configuration data. The configuration data is validated
    against a schema defined in the `config_schema` class attribute. This allows for easy creation of objects with a
    consistent and well-defined interface, while also making it easy to modify the configuration data without changing
    the code that uses it.

    Example:
    >>> class MyCustomizable(Customizable):
    ...     config_schema = {'name': {'type': str, 'optional': True, 'aliases': []},
    ...                      'value': {'type': int, 'default': 0}}
    ...
    >>> config_data = {'name': 'my_object', 'value': 42}
    >>> obj = MyCustomizable.from_config(config_data)
    >>> print(obj.name)
    'my_object'
    >>> print(obj.value)
    42

    In this example, `MyCustomizable` is a subclass of `Customizable`. The `config_schema` attribute defines the
    expected structure and types of the configuration data. The `from_config` method is used to create an instance of
    `MyCustomizable` from the configuration data. The `name` and `value` attributes of the object are set based on the
    configuration data.

    The `config_schema` attribute is a dictionary that maps attribute names to dictionaries that define the expected
    type and other properties of the attribute. The `type` key is required and specifies the expected type of the
    attribute. The `optional` key is optional and specifies whether the attribute is required or not. If `optional` is
    `False` (the default), the attribute must be present in the configuration data. The `aliases` key is optional and
    specifies a list of alternative names that can be used for the attribute in the configuration data. The `default`
    key is optional and specifies a default value for the attribute if it is not present in the configuration data.

    config_schema options are:
    - type: The expected type of the attribute.
    - optional: Whether the attribute is optional or required. Default is False.
    - aliases: A list of alternative names for the attribute in the configuration data. Default is an empty list.

    """
    config_schema = {'name': {'type': str, 'optional': True, 'aliases': []}}

    aliases = []

    @classmethod
    def from_config(cls, config_data, *args, **kwargs):
        """
        Create an instance of the class from configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.

        Returns:
            instance: An instance of the class with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(config_data)
        cls._validate_config(config_data)
        return create_full_instance(cls, config_data, *args, **kwargs)

    def __str__(self):
        def recursive_str(d, indent=0):
            string = ""
            for key, value in d.items():
                if not isinstance(value, GlobalConfig):
                    if isinstance(value, dict):
                        string += f"{' ' * indent}{key}:\n{recursive_str(value, indent + 2)}"
                    else:
                        string += f"{' ' * indent}{key}: {value}\n"
            return string

        config_string = ""
        config_string += recursive_str(self.__dict__)
        return config_string

    @classmethod
    def _preconditions(cls):
        """
        Check if all preconditions are met before running the algorithm.
        """
        pass

    @staticmethod
    def _safe_open(config_data):
        """
        Open and load configuration data from a YAML file or return the provided dictionary.
        """
        if not isinstance(config_data, (str, dict)):
            raise TypeError("Invalid type for config_data. Expected str (file path) or dict.")

        if isinstance(config_data, str):
            try:
                with open(config_data, 'r') as file:
                    config_data = yaml.safe_load(file)
            except Exception as e:
                raise IOError(f"Error loading config file: {e}")

        if not isinstance(config_data, dict):
            raise TypeError("Invalid type for config_data. Expected dict after loading from YAML.")

        return config_data

    @classmethod
    def _validate_config(cls, config_data):
        """
        Check if the configuration data contains all required keys and no invalid keys, and validate the types of the values.
        """
        # Merge config_schema attributes from parent classes
        config_schema = cls.config_schema
        for base in cls.__mro__:
            if hasattr(base, 'config_schema'):
                config_schema.update(base.config_schema)

        # Check for required keys
        missing_keys = []
        for key, value in config_schema.items():
            if 'default' not in value and not value.get('optional', False):
                if key not in config_data and all(alias not in config_data for alias in value.get('aliases', [])):
                    missing_keys.append(key)
        if missing_keys:
            raise KeyError(f"Missing required keys for class {cls.__name__}: {', '.join(missing_keys)}")

        # Check for invalid keys
        invalid_keys = set(config_data.keys()) - set(list(itertools.chain.from_iterable(
            [[key] + v.get('aliases', []) for key, v in config_schema.items()]))) - set(cls.__dict__)
        if invalid_keys:
            warnings.warn(f"Supplementary keys in configuration for class {cls.__name__}: {', '.join(invalid_keys)}")

        # Check for correct types
        for key, schema in config_schema.items():
            aliases = schema.get('aliases', [])
            if key in config_data or any(alias in config_data for alias in aliases):
                value = config_data.get(key, config_data.get(aliases[0]) if aliases else None)
                expected_type = schema.get('type', type(value))
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Invalid type for key '{key}' in configuration for class {cls.__name__}. Expected {expected_type}, got {type(value)}.")
                config_data[key] = value
                for alias in aliases:
                    if alias in config_data:
                        del config_data[alias]

        # Set default values for missing keys
        for key, schema in config_schema.items():
            if key not in config_data and (
                    schema.get('required', False) is True or 'default' in schema) and 'default' in schema and all(
                alias not in config_data for alias in schema.get('aliases', [])):
                config_data[key] = schema.get('default')
        print(config_data)
        cls._preconditions()

    def to_config(self, exclude=[], add={}, parents=True):
        """
        Return a dictionary representation of the instance.
        """
        config = {}
        for key, value in self.__dict__.items():
            if key not in exclude:
                config[key] = value
        config.update(add)
        return config


class TypedCustomizable(Customizable):
    """
        Base class for typed Customizable objects.

        This class is similar to `Customizable`, but it also supports creating objects of different types based on the
        'type' key in the configuration data. This allows for easy creation of objects with a polymorphic interface, where
        the exact type of the object is determined at runtime based on the configuration data.

        Example:
        >>> class MyTypedCustomizable(TypedCustomizable):
        ...     config_schema = {'name': {'type': str, 'optional': True, 'aliases': []},
        ...                      'type': {'type': str}}
        ...
        >>> class MySubclass1(MyTypedCustomizable):
        ...     pass
        ...
        >>> class MySubclass2(MyTypedCustomizable):
        ...     pass
        ...
        >>> config_data1 = {'name': 'my_object1', 'type': 'MySubclass1'}
        >>> obj1 = MyTypedCustomizable.from_config(config_data1)
        >>> print(obj1.name)
        'my_object1'
        >>> print(type(obj1))
        <class '__main__.MySubclass1'>
        >>> config_data2 = {'name': 'my_object2', 'type': 'MySubclass2'}
        >>> obj2 = MyTypedCustomizable.from_config(config_data2)
        >>> print(obj2.name)
        'my_object2'
        >>> print(type(obj2))
        <class '__main__.MySubclass2'>

        In this example, `MyTypedCustomizable` is a subclass of `TypedCustomizable`. The `config_schema` attribute defines
        the expected structure and types of the configuration data, including the required 'type' key. The `from_config`
        method is used to create an instance of `MyTypedCustomizable` from the configuration data. The exact type of the
        object is determined based on the 'type' key in the configuration data.

        The `TypedCustomizable` class provides all the features of `Customizable`, plus the ability to create objects of
        different types based on the configuration data. This makes it easy to create modular and flexible code that can be
        easily extended with new types of objects.
        """

    config_schema = {'name': {'type': str, 'optional': True, 'aliases': []},
                     'type': {'type': str}}

    @classmethod
    def from_config(cls, config_data, *args, **kwargs):
        """
        Create an instance of a subclass from typed configuration data.

        This method finds the correct subclass based on the 'type' key in the configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.

        Returns:
            instance: An instance of the correct subclass with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(config_data)
        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {config_data}")

        def find_subclass_recursive(parent_cls):
            """
            Recursively search for the correct subclass based on the 'type' key.
            """
            for subclass in parent_cls.__subclasses__() + [parent_cls]:
                if type_name.lower() in [alias.lower() for alias in subclass.aliases] + [subclass.__name__.lower()]:
                    subclass._validate_config(config_data)
                    return create_full_instance(subclass, config_data, *args, **kwargs)
                if subclass != parent_cls:
                    recursive_result = find_subclass_recursive(subclass)
                    if recursive_result:
                        return recursive_result
            return None

        result = find_subclass_recursive(cls)

        if result is not None:
            return result
        else:
            subclasses = get_all_subclasses(cls)
            raise Exception(f"Type {type_name} not found. Please check the configuration file. "
                            f"Available types: {[el.__name__ for el in subclasses]}")

def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses

# region Utils

def load_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


def create_full_instance(cls, config_data, *args, **kwargs):
    """
    Create an instance of a class and set its attributes based on the configuration data.
    Args:
        cls (type): The class to create an instance of.
        config_data (dict): The configuration data to use to set the attributes of the instance.
        *args: Additional positional arguments to pass to the class constructor.
        **kwargs: Additional keyword arguments to pass to the class constructor.
    Returns:
        instance: An instance of the class with attributes set according to the configuration data.
    """
    instance = cls.__new__(cls)
    for key, value in config_data.items():
        if not hasattr(instance, key):
            setattr(instance, key, value)
    setattr(instance, 'global_config', GlobalConfig())
    instance.__init__(*args, **kwargs)
    return instance

# endregion
