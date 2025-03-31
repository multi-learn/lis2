from typing import List, Literal

import pytest
from configurable import Configurable, Schema, TypedConfigurable, ValidationError


# Test cases for the Configurable class with complex subclassing

class BaseConfigurable(Configurable):
    config_schema = {
        'base_param': Schema(int, default=0),
    }

    def __init__(self, base_param):
        self.base_param = base_param


class IntermediateConfigurable(BaseConfigurable):
    config_schema = {
        'intermediate_param': Schema(str, default='intermediate'),
    }

    def __init__(self, base_param, intermediate_param):
        super().__init__(base_param)
        self.intermediate_param = intermediate_param


class AdvancedConfigurable(IntermediateConfigurable):
    config_schema = {
        'advanced_param': Schema(float, default=1.0),
    }

    def __init__(self, base_param, intermediate_param, advanced_param = 500000.3):
        super().__init__(base_param, intermediate_param)
        self.advanced_param = advanced_param


def test_advanced_Configurable_from_config():
    config = {
        'base_param': 10,
        'intermediate_param': 'test',
        'advanced_param': 3.14,
    }
    obj = AdvancedConfigurable.from_config(config)
    assert obj.base_param == 10
    assert obj.intermediate_param == 'test'
    assert obj.advanced_param == 3.14


def test_advanced_Configurable_defaults():
    config = {}
    obj = AdvancedConfigurable.from_config(config)
    assert obj.base_param == 0
    assert obj.intermediate_param == 'intermediate'
    assert obj.advanced_param == 1.0


# Test cases with multiple inheritance

class MixinConfigurable(Configurable):
    config_schema = {
        'mixin_param': Schema(bool, default=True),
    }

    def __init__(self, mixin_param = True):
        self.mixin_param = mixin_param


class ComplexConfigurable(AdvancedConfigurable, MixinConfigurable):
    config_schema = {
        'complex_param': Schema(list, default=[]),
    }

    def __init__(self, base_param, intermediate_param, advanced_param, mixin_param, complex_param = None):
        AdvancedConfigurable.__init__(self, base_param, intermediate_param, advanced_param)
        MixinConfigurable.__init__(self, mixin_param)
        self.complex_param = complex_param


def test_complex_Configurable_from_config():
    config = {
        'base_param': 5,
        'intermediate_param': 'inter',
        'advanced_param': 2.5,
        'mixin_param': False,
        'complex_param': [1, 2, 3],
    }
    obj = ComplexConfigurable.from_config(config)
    assert obj.base_param == 5
    assert obj.intermediate_param == 'inter'
    assert obj.advanced_param == 2.5
    assert obj.mixin_param is False
    assert obj.complex_param == [1, 2, 3]


def test_complex_Configurable_defaults():
    config = {}
    obj = ComplexConfigurable.from_config(config)
    assert obj.base_param == 0
    assert obj.intermediate_param == 'intermediate'
    assert obj.advanced_param == 1.0
    assert obj.mixin_param is True
    assert obj.complex_param == []


def test_complex_Configurable_invalid_param():
    config = {
        'base_param': 'not an int',
    }
    with pytest.raises(ValidationError):
        ComplexConfigurable.from_config(config)


# Test cases for errors in class definitions

# Invalid Schema definition (wrong type)
class InvalidSchemaConfigurable(Configurable):
    config_schema = {
        'invalid_param': 'not a Schema instance',
    }


def test_invalid_schema_Configurable():
    config = {'invalid_param': 10}
    with pytest.raises(TypeError):
        InvalidSchemaConfigurable.from_config(config)


# TypedConfigurable with missing 'type' in subclass's config_schema
class MissingTypeConfigurable(TypedConfigurable):
    aliases = ['missing_type']

    # Intentionally omitting 'type' from config_schema
    config_schema = {
        'param': Schema(int, default=1),
    }


def test_missing_type_Configurable():
    config = {'type': 'missing_type', 'param': 2}
    with pytest.raises(ValueError):
        MissingTypeConfigurable.from_config(config)


# TypedConfigurable subclass with conflicting aliases
class ConflictingAliasA(TypedConfigurable):
    aliases = ['conflict']
    config_schema = {
        'type': Schema(str),
        'param_a': Schema(int, default=1),
    }

    def __init__(self, param_a):
        self.param_a = param_a


class ConflictingAliasB(TypedConfigurable):
    aliases = ['conflict']  # Same alias as ConflictingAliasA
    config_schema = {
        'type': Schema(str),
        'param_b': Schema(int, default=2),
    }

    def __init__(self, param_b):
        self.param_b = param_b


# Test for recursive subclass detection
class RecursiveAlgorithm(TypedConfigurable):
    aliases = ['recursive']

    config_schema = {
        'type': Schema(str),
        'param_recursive': Schema(int, default=0),
    }

    def __init__(self, param_recursive):
        self.param_recursive = param_recursive


class SubRecursiveAlgorithm(RecursiveAlgorithm):
    aliases = ['sub_recursive']

    config_schema = {
        'type': Schema(str),
        'param_sub_recursive': Schema(int, default=1),
    }

    def __init__(self, param_recursive, param_sub_recursive):
        super().__init__(param_recursive)
        self.param_sub_recursive = param_sub_recursive


def test_recursive_algorithm():
    config = {'type': 'sub_recursive', 'param_recursive': 5, 'param_sub_recursive': 10}
    obj = TypedConfigurable.from_config(config)
    assert isinstance(obj, SubRecursiveAlgorithm)
    assert obj.param_recursive == 5
    assert obj.param_sub_recursive == 10


def test_recursive_algorithm_defaults():
    config = {'type': 'sub_recursive'}
    obj = TypedConfigurable.from_config(config)
    assert isinstance(obj, SubRecursiveAlgorithm)
    assert obj.param_recursive == 0
    assert obj.param_sub_recursive == 1


# Edge Case: Passing None as configuration data

def test_Configurable_with_none_config():
    with pytest.raises(TypeError):
        Configurable.from_config(None)


def test_typed_Configurable_with_none_config():
    with pytest.raises(TypeError):
        TypedConfigurable.from_config(None)


# Edge Case: Using complex types in Schema (List[int])

class ListConfigurable(Configurable):
    config_schema = {
        'numbers': Schema(List[int]),
    }

    def __init__(self, numbers):
        self.numbers = numbers


def test_list_Configurable_valid():
    config = {'numbers': [1, 2, 3]}
    obj = ListConfigurable.from_config(config)
    assert obj.numbers == [1, 2, 3]


def test_list_Configurable_invalid():
    config = {'numbers': [1, 'two', 3]}
    with pytest.raises(ValidationError):
        ListConfigurable.from_config(config)


# Edge Case: Using custom classes in Schema

class CustomType:
    def __init__(self, value: int):
        self.value = value


class CustomTypeConfigurable(Configurable):
    config_schema = {
        'custom': Schema(CustomType),
    }

    def __init__(self, custom):
        self.custom = custom


def test_custom_type_Configurable_valid():
    custom_obj = CustomType(10)
    config = {'custom': custom_obj}
    obj = CustomTypeConfigurable.from_config(config)
    assert obj.custom.value == 10


def test_custom_type_Configurable_invalid():
    config = {'custom': 'not a CustomType instance'}
    with pytest.raises(ValidationError):
        CustomTypeConfigurable.from_config(config)


# Edge Case: Testing Literal types in Schema

class LiteralConfigurable(Configurable):
    config_schema = {
        'mode': Schema(Literal['train', 'test', 'validate']),
    }

    def __init__(self, mode):
        self.mode = mode


def test_literal_Configurable_valid():
    config = {'mode': 'train'}
    obj = LiteralConfigurable.from_config(config)
    assert obj.mode == 'train'


def test_literal_Configurable_invalid():
    config = {'mode': 'deploy'}
    with pytest.raises(ValidationError):
        LiteralConfigurable.from_config(config)
