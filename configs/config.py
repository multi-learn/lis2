import inspect
import warnings
from functools import wraps
from typing import (
    Union,
    Iterable,
)

import yaml
from typing_extensions import Literal

from utils.logger import setup_logger

"""
author: Julien Rabault
Configuration Management Module for AI Applications.

This module provides classes and utilities for managing configurations,
validating schemas, and creating customizable objects from configuration data.
It is particularly useful for AI applications where configurations can be complex
and need to be validated at runtime.

Classes:
    - Schema: Defines the schema for configuration attributes.
    - GlobalConfig: Singleton class for global configuration settings.
    - Customizable: Base class for objects that can be customized via configuration.
    - TypedCustomizable: Base class for typed customizable objects.

Functions:
    - get_all_subclasses: Recursively retrieves all subclasses of a class.
    - load_yaml: Loads YAML configuration files.

Example:
    ```python
    class MyModel(Customizable):
        config_schema = {
            'learning_rate': Schema(float, default=0.001),
            'epochs': Schema(int, default=10),
        }

        def __init__(self, learning_rate, epochs):
            self.learning_rate = learning_rate
            self.epochs = epochs

    config = {
        'learning_rate': 0.01,
        'epochs': 20,
    }

    model = MyModel.from_config(config)
    ```
"""


class ValidationError(Exception):
    """
    Custom exception class for validation errors.
    """

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or []

    def __str__(self):
        error_messages = "\n".join(self.errors)
        return f"{self.args[0]}\n{error_messages}"


Config = Union[dict, str]

from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin, Literal
import collections.abc


class Schema:
    """
    Defines the schema for configuration attributes.

    Attributes:
        expected_type (Type): The expected type of the configuration attribute.
        aliases (List[str], optional): Alternative keys for the configuration attribute.
        optional (bool, optional): Indicates whether the configuration attribute is optional.
        default (Any, optional): Default value for the configuration attribute if it is missing.

    Examples:
        ```python
        Schema(int, aliases=['num_epochs'], optional=True, default=10)
        ```
    """

    def __init__(
        self,
        type: Type,
        aliases: Optional[List[str]] = None,
        optional: bool = False,
        default: Any = None,
    ):
        """
        Initializes a Schema instance.

        Args:
            expected_type (Type): The expected type of the configuration attribute.
            aliases (List[str], optional): Alternative keys for the configuration attribute.
            optional (bool, optional): Indicates whether the configuration attribute is optional.
            default (Any, optional): Default value for the configuration attribute if it is missing.
        """
        self.expected_type = type
        self.aliases = aliases or []
        self.optional = optional or default is not None
        self.default = default

    def validate(self, config: Dict[str, Any], key: str) -> Any:
        """
        Validates and retrieves the value of a configuration attribute from a config dictionary.

        Args:
            config (Dict[str, Any]): The configuration dictionary to validate.
            key (str): The primary key for the configuration attribute.

        Returns:
            Any: The validated and possibly converted value of the configuration attribute.

        Raises:
            ValueError: If the value is missing (and not optional), or if the value cannot be converted to expected type.
        """
        keys_to_check = [key] + self.aliases
        for k in keys_to_check:
            if k in config:
                value = config[k]
                return self._validate_type(value, self.expected_type)
        if self.optional:
            return self.default
        else:
            raise KeyError(
                f"Required configuration key(s) {keys_to_check} not found in config."
            )

    def _validate_type(self, value: Any, expected_type: Type) -> Any:
        """
        Recursively validates the value against the expected type.

        Args:
            value (Any): The value to validate.
            expected_type (Type): The expected type.

        Returns:
            Any: The validated value.

        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        if origin is Union:
            # Try each type in the Union
            for typ in args:
                try:
                    return self._validate_type(value, typ)
                except TypeError:
                    continue
            expected_types = ", ".join(self._type_name(t) for t in args)
            raise TypeError(
                f"Value '{value}' does not match any type in Union[{expected_types}]"
            )
        elif origin is Literal:
            if value in args:
                return value
            else:
                raise TypeError(f"Value '{value}' is not a valid Literal {args}")
        elif origin in (list, List):
            if not isinstance(value, list):
                raise TypeError(f"Expected list but got {type(value).__name__}")
            if not args:
                return value  # No type specified for list elements
            element_type = args[0]
            return [self._validate_type(item, element_type) for item in value]
        elif origin in (dict, Dict):
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict but got {type(value).__name__}")
            if not args or len(args) != 2:
                return value  # No type specified for dict keys and values
            key_type, val_type = args
            return {
                self._validate_type(k, key_type): self._validate_type(v, val_type)
                for k, v in value.items()
            }
        elif origin in (Iterable, collections.abc.Iterable):
            if not isinstance(value, collections.abc.Iterable):
                raise ValidationError(
                    f"Expected iterable but got {type(value).__name__}"
                )
            if not args:
                return value
            element_type = args[0]
            return type(value)(
                self._validate_type(item, element_type) for item in value
            )
        elif isinstance(expected_type, type):
            if isinstance(value, expected_type):
                return value
            else:
                raise TypeError(
                    f"Expected type {expected_type.__name__} but got {type(value).__name__}"
                )
        else:
            raise TypeError(f"Unsupported type {expected_type}")

    def _type_name(self, typ: Type) -> str:
        """
        Retrieves a string representation of the type.

        Returns:
            str: The name of the type.
        """
        origin = get_origin(typ)
        args = get_args(typ)
        if origin is Union:
            return f"Union[{', '.join(self._type_name(t) for t in args)}]"
        elif origin is Literal:
            return f"Literal{args}"
        elif origin in (list, List):
            if args:
                return f"List[{self._type_name(args[0])}]"
            else:
                return "List"
        elif origin in (dict, Dict):
            if args and len(args) == 2:
                return f"Dict[{self._type_name(args[0])}, {self._type_name(args[1])}]"
            else:
                return "Dict"
        elif origin in (Iterable, collections.abc.Iterable):
            if args:
                return f"Iterable[{self._type_name(args[0])}]"
            else:
                return "Iterable"
        elif hasattr(typ, "__name__"):
            return typ.__name__
        else:
            return str(typ)

    def __repr__(self):
        return f"Schema(type={self.expected_type}, aliases={self.aliases}, optional={self.optional}, default={self.default})"


class GlobalConfig:
    """
    Singleton class that holds global configuration data.

    This class ensures that only one instance of the global configuration exists,
    which can be accessed and modified throughout the application.

    Attributes:
        _instance (GlobalConfig): The singleton instance.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        if config is not None:
            self.__dict__.update(self._process_config(config))

    def _process_config(self, config):
        processed_config = {}
        for key, value in config.items():
            if isinstance(value, str) and value.endswith('.yml'):
                try:
                    with open(value, 'r') as yml_file:
                        processed_config[key] = yaml.safe_load(yml_file)
                except Exception as e:
                    raise ValueError(f"Error loading YAML file '{value}': {e}")
            else:
                processed_config[key] = value
        return processed_config

    def __setitem__(self, name, value):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        self.__dict__.setdefault(name, None)
        self.__dict__[name] = value

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        if name not in self.__dict__:
            raise KeyError(
                f"GlobalConfig does not have key: {name}, see: {self.__dict__}"
            )
        return self.__dict__.get(name, None)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        return self.__dict__


class Customizable:
    """
    Base class for customizable objects using the `from_config` class method.

    This class allows objects to be created and configured from a configuration dictionary or file.

    Attributes:
        config_schema (dict): Defines the schema for configuration attributes.
        aliases (list): Alternative names for the class, useful for subclass identification.

    Example:
        ```python
        class MyAlgorithm(Customizable):
            config_schema = {
                'learning_rate': Schema(float, optional=True, default=0.01),
                'batch_size': Schema(int, optional=True, default=32),
            }

            def __init__(self, learning_rate, batch_size):
                self.learning_rate = learning_rate
                self.batch_size = batch_size

        config = {
            'learning_rate': 0.001,
            'batch_size': 64,
        }

        algorithm = MyAlgorithm.from_config(config)
        ```
    """

    config_schema = {"name": Schema(Union[str, None], optional=True, default=None)}
    aliases = []

    @classmethod
    def from_config(cls, config_data, *args, debug=False, **kwargs):
        """
        Creates an instance of the class from configuration data.

        Args:
            config_data (dict or str): Configuration data as a dictionary or a path to a YAML file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Customizable: An instance of the class.

        Raises:
            IOError: If there is an error loading the configuration file.
            TypeError: If the configuration data is of invalid type.
            KeyError: If the configuration data is missing required keys.
            ValidationError: If there are validation errors in the configuration data.
        """
        return cls._from_config(config_data, *args, debug=debug, **kwargs)

    @classmethod
    def _from_config(cls, config_data, *args, debug=False, **kwargs):
        """
        Core logic for creating an instance from configuration data.

        Args:
            config_data (dict or str): Configuration data as a dictionary or a path to a YAML file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Customizable: An instance of the class.
        """
        config_data = cls._safe_open(config_data)
        config_validate = cls._validate_config(config_data)

        # Create a new subclass with a wrapped __init__
        WrappedClass = cls._create_wrapped_class(config_validate, debug=debug)

        # Instantiate the wrapped class
        instance = WrappedClass(*args, **kwargs)
        return instance

    @classmethod
    def _create_wrapped_class(cls, config_validate, debug=False):
        """
        Creates a new subclass with a wrapped __init__ method that sets configuration attributes.

        Args:
            config_validate (dict): The validated configuration data. This dictionary contains
                                    configuration key-value pairs to be set as attributes on the class instance.
            debug (bool): If True, enables debugging mode for the logger. Defaults to False.

        Returns:
            type: A new dynamically created class that inherits from the current class (cls).
        """
        # Save the original __init__ method of the class to be wrapped later
        original_init = cls.__init__

        @wraps(original_init)  # Preserve metadata of the original __init__ method
        def wrapped_init(self, *args, **kwargs):
            """
            A replacement for the __init__ method of the class. This method:
            - Sets configuration attributes.
            - Initializes a logger.
            - Ensures required arguments for the original __init__ method are set.
            """

            # Set attributes from the validated configuration
            for key, value in config_validate.items():
                setattr(self, key, value)

            # Initialize global and configuration-specific settings
            self.global_config = GlobalConfig()
            self.config = config_validate

            # Generate the logger name using the class name and optional instance-specific name
            name = self.__class__.__name__ + f"[{self.name}]" if self.name else self.__class__.__name__
            self.logger = setup_logger(name, config_validate, debug=debug)

            # Retrieve the signature of the original __init__ method
            init_signature = inspect.signature(original_init)
            init_params = init_signature.parameters

            # Filter parameters to exclude 'self', '*args', and '**kwargs'
            init_params = {
                k: v
                for k, v in init_params.items()
                if k != "self" and k != "args" and k != "kwargs"
            }

            # Collect arguments for the original __init__ method
            init_args = {}
            for name, param in init_params.items():
                if name in kwargs:
                    # Use the value from kwargs if provided
                    init_args[name] = kwargs.pop(name)
                elif name in config_validate:
                    # Use the value from the validated configuration
                    init_args[name] = config_validate[name]
                elif param.default != inspect.Parameter.empty:
                    # Use the default value from the original __init__ signature if available
                    pass
                else:
                    # Raise an error if a required argument is missing
                    raise TypeError(
                        f"Missing required argument '{name}' for {cls.__name__}.__init__"
                    )

            # Perform any required checks or operations before initialization
            self.preconditions()

            # Call the original __init__ method with the prepared arguments
            original_init(self, *args, **init_args)

        # Dynamically create a new class inheriting from the original class (cls)
        WrappedClass = type(cls.__name__, (cls,), {"__init__": wrapped_init})

        # Return the new wrapped class
        return WrappedClass

    @classmethod
    def _safe_open(cls, config_data):
        if isinstance(config_data, str):
            try:
                with open(config_data, "r") as file:
                    config_data = yaml.safe_load(file)
            except Exception as e:
                raise IOError(f"Error loading config file: {e}")
        elif not isinstance(config_data, dict):
            raise TypeError(
                "Invalid type for config_data. Expected dict after loading from YAML."
            )
        return config_data

    @classmethod
    def _validate_config(cls, config_data, dynamic_schema=None):
        if dynamic_schema is None:
            dynamic_schema = {}
        config_schema = {}
        # Collect config_schema from all bases
        for base in reversed(cls.__mro__):
            if hasattr(base, "config_schema"):
                config_schema.update(base.config_schema)
        config_schema.update(dynamic_schema)

        validated_config = {}
        errors = []

        for key, schema in config_schema.items():
            if not isinstance(schema, Schema):
                raise TypeError(
                    f"Schema object expected for key '{key}' in class '{cls.__name__}'"
                )

            try:
                validated_value = schema.validate(config_data, key)
                validated_config[key] = validated_value
            except KeyError:
                errors.append(
                    f"Missing required key '{key}' in configuration for class '{cls.__name__}'."
                )
            except TypeError as e:
                errors.append(
                    f"Type error for key '{key}' in class '{cls.__name__}': {str(e)}"
                )
            except ValueError as e:
                errors.append(
                    f"Value error for key '{key}' in class '{cls.__name__}': {str(e)}"
                )
            except Exception as e:
                errors.append(
                    f"Unexpected error for key '{key}' in class '{cls.__name__}': {str(e)}"
                )

        if errors:
            cls_name = cls.__name__
            cls_aliases = ", ".join(cls.aliases)
            cls_name_aliases = (
                f"{cls_name} [{cls_aliases}]" if cls.aliases else cls_name
            )
            raise ValidationError(
                f"Validation errors in configuration for class '{cls_name_aliases}':",
                errors=errors,
            )

        # Check for unexpected keys
        valid_keys = set(config_schema.keys())
        for schema in config_schema.values():
            if isinstance(schema.aliases, list):
                valid_keys.update(schema.aliases)

        invalid_keys = set(config_data.keys()) - valid_keys
        if invalid_keys:
            warnings.warn(
                f"Unknown keys in configuration for class '{cls.__name__}': {', '.join(invalid_keys)}",
                UserWarning,
            )

        return validated_config


    def preconditions(self):
        """
        Check if all preconditions are met before running the algorithm.
        """
        pass

    def to_config(self, exclude=[], add={}):
        config = {}
        for key, value in self.__dict__.items():
            if key not in exclude and not key.startswith("_"):
                config[key] = value
        config.update(add)
        return config

    def get_config_schema(self):
        return self.config_schema

    def __str__(self):
        def recursive_str(d, indent=0):
            string = ""
            for key, value in d.items():
                if not isinstance(value, GlobalConfig):
                    if isinstance(value, dict):
                        string += (
                            f"{' ' * indent}{key}:\n{recursive_str(value, indent + 2)}"
                        )
                    else:
                        string += f"{' ' * indent}{key}: {value}\n"
            return string

        config_string = ""
        config_string += recursive_str(self.__dict__)
        return config_string

    def save_dict_to_yaml(self, data: dict, file_path: str):
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)


class TypedCustomizable(Customizable):
    """
    Base class for typed customizable objects.

    This class extends `Customizable` to allow for dynamic subclass instantiation
    based on a 'type' key in the configuration data.

    Attributes:
        config_schema (dict): Defines the schema for configuration attributes, including 'type'.

    Example:
        ```python
        class BaseModel(TypedCustomizable):
            aliases = ['base_model']

        class CNNModel(BaseModel):
            aliases = ['cnn', 'convolutional']

            config_schema = {
                'filters': Schema(int, default=32),
                'kernel_size': Schema(int, default=3),
            }

            def __init__(self, filters, kernel_size):
                self.filters = filters
                self.kernel_size = kernel_size

        config = {
            'type': 'cnn',
            'filters': 64,
            'kernel_size': 5,
        }

        model = BaseModel.from_config(config)
        ```
    """

    config_schema = {"type": Schema(str)}

    @classmethod
    def from_config(cls, config_data, *args, **kwargs):
        """
        Create an instance of the correct subclass based on 'type' in config_data.
        """
        config_data = cls._safe_open(config_data)
        try:
            type_name = config_data["type"]
        except KeyError:
            raise ValueError(
                f"Missing required key: 'type' for class {cls.__name__} in config file."
            )

        subclass = cls.find_subclass_by_type_name(type_name)
        if subclass is None:
            subclasses = get_all_subclasses(cls)
            raise ValueError(
                f"Type '{type_name}' not found. Available types: {[el.get_all_name() for el in subclasses]}\n"
                f"If you add a custom class in a new files .py, make sure to add it import in the __init__.py file"
            )

        return subclass._from_config(config_data, *args, **kwargs)

    @classmethod
    def find_subclass_by_type_name(cls, type_name: str):
        assert (
            type(type_name) == str
        ), f"type_name must be a string, got {type(type_name)}"
        for subclass in cls.__subclasses__():
            if type_name.lower() in [alias.lower() for alias in subclass.aliases] + [
                subclass.__name__.lower()
            ]:
                return subclass
            else:
                subsubclass = subclass.find_subclass_by_type_name(type_name)
                if subsubclass:
                    return subsubclass
        return None

    @classmethod
    def get_all_name(cls):
        return f"{cls.__name__} ({', '.join(cls.aliases)})"


# region Utility Functions


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def load_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


# endregion
