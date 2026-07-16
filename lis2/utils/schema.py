import inspect
from typing import Any, Type, Union, get_args

from configurable import Schema


def get_primary_type(default_value: Union) -> Any:
    """
    Infer one of the type inside a union of types

    Args:
        default_value (Union): The default value to infer the type from.

    Returns:
        Any: The inferred type.
    """

    type_annotation = get_args(default_value)
    if not type_annotation:
        return default_value

    if int in type_annotation:
        return int
    if float in type_annotation:
        return float
    if bool in type_annotation:
        return bool
    if str in type_annotation:
        return str
    return None


def infer_type_from_annotation(
    annotation_value: Any,
) -> tuple[Type] | list[Type] | Type:
    """
    Infers the type annotation from the annotation.

    Args:
        annotation_value (Any): The default value to infer the type from.

    Returns:
        Any: The inferred type.
    """

    type_annotation = get_args(annotation_value)
    if type_annotation and isinstance(type_annotation, tuple):
        list = []
        for element in type_annotation:
            element_type = get_primary_type(element)
            if element_type is not None and element_type in (int, float, bool, str):
                list.append(element_type)
        return tuple(list)
    return annotation_value


def infer_type_from_default(default_value: Any) -> tuple[Type] | list[Type] | Type:
    """
    Infers the type annotation from the default value.

    Args:
        default_value (Any): The default value to infer the type from.

    Returns:
        Any: The inferred type.
    """
    if isinstance(default_value, tuple):
        element_types = tuple(
            get_primary_type(type(element)) for element in default_value
        )
        return element_types
    elif isinstance(default_value, list):
        element_type = (
            get_primary_type(type(default_value[0])) if default_value else Any
        )
        return list[element_type]
    elif type(default_value) == Union:
        return get_primary_type(default_value)
    else:
        return type(default_value)


def generate_config_schema(input_class: Type[Any]) -> dict[str, Schema]:
    """
    Automatically generates a configuration schema for a given optimizer
    by inspecting its __init__ parameters.

    Args:
        input_class (Type[Optimizer]): The input class to generate the schema for.

    Returns:
        dict[str, Schema]: A dictionary mapping parameter names to their corresponding schema.
    """
    config_schema = {}
    init_signature = inspect.signature(input_class.__init__)
    for param_name, param in init_signature.parameters.items():
        if param_name in ["self", "params", "defaults"]:
            continue

        optional = param.default != inspect.Parameter.empty
        default = param.default if optional else None

        if param.annotation != inspect.Parameter.empty:
            param_type = infer_type_from_annotation(param.annotation)
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
