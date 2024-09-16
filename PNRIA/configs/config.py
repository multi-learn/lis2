import logging
import yaml

from PNRIA.utils.distributed import get_rank_num


class GlobalConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GlobalConfig, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, config=None):
        if config is not None:
            self.__dict__.update(config.copy())

    def __setitem__(self, name, value):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        self.__dict__.setdefault(name, None)
        self.__dict__[name] = value

    def __getitem__(self, name):
        if not isinstance(name, str):
            raise TypeError("GlobalConfig keys must be strings")
        if name not in self.__dict__:
            raise KeyError(f"GlobalConfig does not have key: {name},\n see :{self.__dict__}")
        return self.__dict__.get(name, None)

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

    def to_dict(self):
        return self.__dict__


class Configurable:
    """
    Base class for configurable objects.

    This class provides methods to load and validate configuration data from a YAML file or a dictionary. Subclasses
    should define `required_keys` and `aliases` class attributes. When a subclass is initialized with configuration
    data, the attributes defined in the configuration will be automatically set on the instance.
    """
    required_keys = []

    # aliases for the class name
    aliases = []

    logger = logging.getLogger(f"log_{get_rank_num()}")

    @classmethod
    def from_config(cls, config_data, *args, **kwargs):
        """of
        Create an instance  the class from configuration data.
        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.
            **kwargs: Additional keyword arguments to pass to the class constructor.
        Returns:
            instance: An instance of the class with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(config_data)
        cls._check_config(config_data)
        return create_full_instance(cls, config_data, *args, **kwargs)

    @classmethod
    def from_typed_config(cls, config_data, *args, **kwargs):
        """
        Create an instance of a subclass from typed configuration data.

        This method finds the correct subclass based on the 'type' key in the configuration data.

        Args:
            config_data (str or dict): Configuration data in the form of a dictionary or a path to a YAML file.
            **kwargs: Additional keyword arguments to pass to the class constructor.
        Returns:
            instance: An instance of the correct subclass with attributes set according to the configuration data.
        """
        config_data = cls._safe_open(config_data)
        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        def find_subclass_recursive(parent_cls):
            """
            Recursively search for the correct subclass based on the 'type' key.
            """
            for subclass in parent_cls.__subclasses__() + [parent_cls]:
                if type_name in subclass.aliases + [subclass.__name__]:
                    subclass._check_config(config_data, typed=True)
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
            raise Exception(f"Type {type_name} not found. Please check the configuration file. "
                f"Available types: {[el.__name__ for el in cls.__subclasses__()]}")

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
    def _check_config(cls, config_data, typed=False, dynamic_keys=None):
        """
        Check if the configuration data contains all required keys and no invalid keys.
        Args:
            config_data (dict): Configuration data to check.
            typed (bool): Whether the configuration data is typed.
            dynamic_keys (list): List of dynamic keys to add to the required keys.
        """
        required_keys = []
        if typed:
            required_keys = cls.required_keys + ['type']
        current_class = cls

        # Add required keys from all parent classes
        while hasattr(current_class, 'required_keys'):
            required_keys += current_class.required_keys
            current_class = current_class.__base__

        # Add dynamic keys if provided, dynamic keys are keys that are not known in advance
        # useful for configuration files that can have different keys depending on the context or not custom class
        if dynamic_keys is not None:
            required_keys += dynamic_keys
        # unique required keys
        required_keys = list(set(required_keys))

        invalid_keys = set(config_data.keys()) - set(required_keys) - set(cls.__dict__)
        if invalid_keys:
            Warning(f"Supplementary keys in configuration for class {cls.__name__}: {', '.join(invalid_keys)}")

        missing_keys = [key for key in required_keys if key not in config_data]
        if missing_keys:
            raise ValueError(f"Missing required keys for class {cls.__name__}: {', '.join(missing_keys)}")

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


# region Utils

def load_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data


def create_full_instance(cls, config_data, *args, **kwargs):
    instance = cls.__new__(cls)
    for key, value in config_data.items():
        if not hasattr(instance, key):
            setattr(instance, key, value)
    setattr(instance, 'global_config', GlobalConfig())
    instance.__init__(*args, **kwargs)
    return instance

# endregion
