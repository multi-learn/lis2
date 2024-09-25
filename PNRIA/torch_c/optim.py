import inspect
import logging
import torch.optim as optim
from PNRIA.configs.config import GlobalConfig, TypedCustomizable, Schema


def get_optimizer_and_schema(type_name):
    optimizer_dict = get_all_optimizers()
    if type_name in optimizer_dict:
        return optimizer_dict[type_name]
    else:
        return None, None


def generate_config_schema(optimizer_class):
    config_schema = {}
    init_signature = inspect.signature(optimizer_class.__init__)
    for param in init_signature.parameters.values():
        if param.name not in ["self", "params", "defaults"]:
            config_schema[param.name] = Schema(
                type=param.annotation if param.annotation != param.empty else type(param.default) if param.default != param.empty else str,
                optional=param.default != param.empty,
                default=param.default if param.default != param.empty else None,
            )
    return config_schema


def get_all_optimizers():
    optimizer_dict = {}

    # Ajout des optimizers torch
    for name, obj in optim.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, optim.Optimizer) and obj is not optim.Optimizer:
            optimizer_dict[name] = (obj, generate_config_schema(obj))

    # Ajout des optimizers personnalisés
    for subclass in BaseOptimizer.__subclasses__():
        optimizer_dict[subclass.__name__] = (subclass, subclass.get_config_schema())
    return optimizer_dict


class BaseOptimizer(TypedCustomizable, optim.Optimizer):
    """Base class pour les optimizers avec une configuration typée."""


    @classmethod
    def from_config(cls, config_data, params, **kwargs):
        """Crée une instance d'optimizer à partir des données de configuration."""
        config_data = cls._safe_open(config_data)

        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        optimizer_class, config_schema = get_optimizer_and_schema(type_name)
        if optimizer_class is None:
            raise Exception(f"Type {type_name} not found, please check the configuration file. "
                            f"List of available types: {[el.__name__ for el in cls.__subclasses__()]}")

        # Vérification de la configuration
        cls._validate_config(config_data, dynamic_schema=config_schema)

        # Création de l'instance de l'optimizer
        instance = optimizer_class.__new__(optimizer_class)
        for key, value in config_data.items():
            if not hasattr(instance, key):
                setattr(instance, key, value)

        # Ajout de la configuration globale
        setattr(instance, 'global_config', GlobalConfig())
        # pop type key
        config_data.pop('type', None)
        instance.__init__(params, **config_data, **kwargs)
        return instance
