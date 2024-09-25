import logging

import torch
import inspect

from torch.optim import lr_scheduler

from PNRIA.configs.config import TypedCustomizable, GlobalConfig, Schema

EXCLUDE_SCHEDULERS = ["_LRScheduler", "LRScheduler", "SequentialLR"]


def generate_config_schema(scheduler_class):
    config_schema = {}
    init_signature = inspect.signature(scheduler_class.__init__)
    for param in init_signature.parameters.values():
        if param.name not in ["self", "params", "defaults"]:
            config_schema[param.name] = Schema(
                type=param.annotation if param.annotation != param.empty else type(param.default) if param.default != param.empty else str,
                optional=param.default != param.empty,
                default=param.default if param.default != param.empty else None,
            )
    return config_schema


def get_all_schedulers(excludes=EXCLUDE_SCHEDULERS):
    scheduler_dict = {}

    # Récupération des schedulers de torch
    schedulers = inspect.getmembers(lr_scheduler, inspect.isclass)
    print(schedulers)
    for name, obj in schedulers:
        if isinstance(obj, type) and issubclass(obj, torch.optim.lr_scheduler.LRScheduler) and name not in excludes:
            scheduler_dict[name] = (obj, generate_config_schema(obj))

    # Ajout des schedulers personnalisés
    for subclass in BaseScheduler.__subclasses__():
        scheduler_dict[subclass.__name__] = (subclass, subclass.get_config_schema())

    return scheduler_dict


def get_scheduler_and_keys(type_name):
    scheduler_dict = get_all_schedulers()
    return scheduler_dict.get(type_name, (None, None))


class BaseScheduler(TypedCustomizable, lr_scheduler._LRScheduler):

    use_step_each_batch_list = ["CyclicLR", "OneCycleLR"]

    @classmethod
    def from_config(cls, config_data, optimizer, **kwargs):
        """Crée une instance de scheduler à partir des données de configuration."""
        config_data = cls._safe_open(config_data)

        try:
            type_name = config_data['type']
        except KeyError:
            raise ValueError(f"Missing required key: type for class {cls.__name__} in config file for {cls.__name__}")

        scheduler_class, config_schema = get_scheduler_and_keys(type_name)
        if scheduler_class is None:
            raise Exception(f"Type {type_name} not found, please check the configuration file. "
                            f"List of available types: {[el.__name__ for el in cls.__subclasses__()]}")

        cls._validate_config(config_data)

        instance = scheduler_class.__new__(scheduler_class)

        config_data.pop('type', None)

        for key, value in config_data.items():
            if not hasattr(instance, key):
                setattr(instance, key, value)

        setattr(instance, 'global_config', GlobalConfig())

        config_data.pop('type', None)
        instance.__init__(optimizer, **config_data, **kwargs)

        if type_name in cls.use_step_each_batch_list:
            setattr(instance, "step_each_batch", True)
        else:
            setattr(instance, "step_each_batch", False)

        return instance
