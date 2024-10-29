from typing import List, Literal

import pytest

from PNRIA.configs.config import Customizable, Schema, TypedCustomizable, ValidationError


# Test cases for the Customizable class with complex subclassing

class BaseCustomizable(Customizable):
    config_schema = {
        'base_param': Schema(int, default=0),
    }

    def __init__(self, base_param):
        self.base_param = base_param


class IntermediateCustomizable(BaseCustomizable):
    config_schema = {
        'intermediate_param': Schema(str, default='intermediate'),
    }

    def __init__(self, base_param, intermediate_param):
        super().__init__(base_param)
        self.intermediate_param = intermediate_param


class AdvancedCustomizable(IntermediateCustomizable):
    config_schema = {
        'advanced_param': Schema(float, default=1.0),
    }

    def __init__(self, base_param, intermediate_param, advanced_param):
        super().__init__(base_param, intermediate_param)
        self.advanced_param = advanced_param


def test_advanced_customizable_from_config():
    config = {
        'base_param': 10,
        'intermediate_param': 'test',
        'advanced_param': 3.14,
    }
    obj = AdvancedCustomizable.from_config(config)
    assert obj.base_param == 10
    assert obj.intermediate_param == 'test'
    assert obj.advanced_param == 3.14


def test_advanced_customizable_defaults():
    config = {}
    obj = AdvancedCustomizable.from_config(config)
    assert obj.base_param == 0
    assert obj.intermediate_param == 'intermediate'
    assert obj.advanced_param == 1.0


# Test cases with multiple inheritance

class MixinCustomizable(Customizable):
    config_schema = {
        'mixin_param': Schema(bool, default=True),
    }

    def __init__(self, mixin_param):
        self.mixin_param = mixin_param


class ComplexCustomizable(AdvancedCustomizable, MixinCustomizable):
    config_schema = {
        'complex_param': Schema(list, default=[]),
    }

    def __init__(self, base_param, intermediate_param, advanced_param, mixin_param, complex_param):
        AdvancedCustomizable.__init__(self, base_param, intermediate_param, advanced_param)
        MixinCustomizable.__init__(self, mixin_param)
        self.complex_param = complex_param


def test_complex_customizable_from_config():
    config = {
        'base_param': 5,
        'intermediate_param': 'inter',
        'advanced_param': 2.5,
        'mixin_param': False,
        'complex_param': [1, 2, 3],
    }
    obj = ComplexCustomizable.from_config(config)
    assert obj.base_param == 5
    assert obj.intermediate_param == 'inter'
    assert obj.advanced_param == 2.5
    assert obj.mixin_param is False
    assert obj.complex_param == [1, 2, 3]


def test_complex_customizable_defaults():
    config = {}
    obj = ComplexCustomizable.from_config(config)
    assert obj.base_param == 0
    assert obj.intermediate_param == 'intermediate'
    assert obj.advanced_param == 1.0
    assert obj.mixin_param is True
    assert obj.complex_param == []


def test_complex_customizable_invalid_param():
    config = {
        'base_param': 'not an int',
    }
    with pytest.raises(ValidationError):
        ComplexCustomizable.from_config(config)


# Test cases for errors in class definitions

# Invalid Schema definition (wrong type)
class InvalidSchemaCustomizable(Customizable):
    config_schema = {
        'invalid_param': 'not a Schema instance',
    }


def test_invalid_schema_customizable():
    config = {'invalid_param': 10}
    with pytest.raises(TypeError):
        InvalidSchemaCustomizable.from_config(config)


# TypedCustomizable with missing 'type' in subclass's config_schema
class MissingTypeCustomizable(TypedCustomizable):
    aliases = ['missing_type']

    # Intentionally omitting 'type' from config_schema
    config_schema = {
        'param': Schema(int, default=1),
    }


def test_missing_type_customizable():
    config = {'type': 'missing_type', 'param': 2}
    with pytest.raises(ValueError):
        MissingTypeCustomizable.from_config(config)


# TypedCustomizable subclass with conflicting aliases
class ConflictingAliasA(TypedCustomizable):
    aliases = ['conflict']
    config_schema = {
        'type': Schema(str),
        'param_a': Schema(int, default=1),
    }

    def __init__(self, param_a):
        self.param_a = param_a


class ConflictingAliasB(TypedCustomizable):
    aliases = ['conflict']  # Same alias as ConflictingAliasA
    config_schema = {
        'type': Schema(str),
        'param_b': Schema(int, default=2),
    }

    def __init__(self, param_b):
        self.param_b = param_b


# Test for recursive subclass detection
class RecursiveAlgorithm(TypedCustomizable):
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
    obj = TypedCustomizable.from_config(config)
    assert isinstance(obj, SubRecursiveAlgorithm)
    assert obj.param_recursive == 5
    assert obj.param_sub_recursive == 10


def test_recursive_algorithm_defaults():
    config = {'type': 'sub_recursive'}
    obj = TypedCustomizable.from_config(config)
    assert isinstance(obj, SubRecursiveAlgorithm)
    assert obj.param_recursive == 0
    assert obj.param_sub_recursive == 1


# Edge Case: Passing None as configuration data

def test_customizable_with_none_config():
    with pytest.raises(TypeError):
        Customizable.from_config(None)


def test_typed_customizable_with_none_config():
    with pytest.raises(TypeError):
        TypedCustomizable.from_config(None)


# Edge Case: Using complex types in Schema (List[int])

class ListCustomizable(Customizable):
    config_schema = {
        'numbers': Schema(List[int]),
    }

    def __init__(self, numbers):
        self.numbers = numbers


def test_list_customizable_valid():
    config = {'numbers': [1, 2, 3]}
    obj = ListCustomizable.from_config(config)
    assert obj.numbers == [1, 2, 3]


def test_list_customizable_invalid():
    config = {'numbers': [1, 'two', 3]}
    with pytest.raises(ValidationError):
        ListCustomizable.from_config(config)


# Edge Case: Using custom classes in Schema

class CustomType:
    def __init__(self, value: int):
        self.value = value


class CustomTypeCustomizable(Customizable):
    config_schema = {
        'custom': Schema(CustomType),
    }

    def __init__(self, custom):
        self.custom = custom


def test_custom_type_customizable_valid():
    custom_obj = CustomType(10)
    config = {'custom': custom_obj}
    obj = CustomTypeCustomizable.from_config(config)
    assert obj.custom.value == 10


def test_custom_type_customizable_invalid():
    config = {'custom': 'not a CustomType instance'}
    with pytest.raises(ValidationError):
        CustomTypeCustomizable.from_config(config)


# Edge Case: Testing Literal types in Schema

class LiteralCustomizable(Customizable):
    config_schema = {
        'mode': Schema(Literal['train', 'test', 'validate']),
    }

    def __init__(self, mode):
        self.mode = mode


def test_literal_customizable_valid():
    config = {'mode': 'train'}
    obj = LiteralCustomizable.from_config(config)
    assert obj.mode == 'train'


def test_literal_customizable_invalid():
    config = {'mode': 'deploy'}
    with pytest.raises(ValidationError):
        LiteralCustomizable.from_config(config)
