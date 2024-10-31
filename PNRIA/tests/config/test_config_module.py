import pytest

from PNRIA.configs.config import Schema, GlobalConfig, Customizable, TypedCustomizable, ValidationError


# Test cases for the Schema class
def test_schema_validation_required_field():
    schema = Schema(int)
    config_data = {'key': 5}
    assert schema.validate(config_data, 'key') == 5


def test_schema_validation_missing_required_field():
    schema = Schema(int)
    config_data = {}
    with pytest.raises(KeyError):
        schema.validate(config_data, 'key')


def test_schema_validation_optional_field_with_default():
    schema = Schema(int, optional=True, default=10)
    config_data = {}
    assert schema.validate(config_data, 'key') == 10


def test_schema_validation_aliases():
    schema = Schema(int, aliases=['alias1', 'alias2'])
    config_data = {'alias2': 15}
    assert schema.validate(config_data, 'key') == 15


def test_schema_validation_invalid_type():
    schema = Schema(int)
    config_data = {'key': 'not an int'}
    with pytest.raises(TypeError):
        schema.validate(config_data, 'key')


# Test cases for the GlobalConfig class
def test_global_config_singleton():
    config1 = GlobalConfig()
    config2 = GlobalConfig()
    assert config1 is config2


def test_global_config_set_and_get_item():
    config = GlobalConfig()
    config['key'] = 'value'
    assert config['key'] == 'value'


def test_global_config_invalid_key_type():
    config = GlobalConfig()
    with pytest.raises(TypeError):
        config[123] = 'value'


def test_global_config_missing_key():
    config = GlobalConfig()
    with pytest.raises(KeyError):
        _ = config['nonexistent']


# Test cases for the Customizable class
class MyCustomizable(Customizable):
    config_schema = {
        'param0': Schema(int),
        'param1': Schema(int, default=1),
        'param2': Schema(str, optional=True),
    }


def test_customizable_from_config():
    config = {'param0': 1, 'param1': 10, 'param2': 'test'}
    obj = MyCustomizable.from_config(config)
    assert obj.param0 == 1
    assert obj.param1 == 10
    assert obj.param2 == 'test'

def test_override_default_value():
    config = {'param0': 1, 'param1': 10}
    obj = MyCustomizable.from_config(config)
    assert obj.param0 == 1
    assert obj.param1 == 10
    assert obj.param2 is None


def test_customizable_from_config_with_defaults():
    config = {'param0': 1, }
    obj = MyCustomizable.from_config(config)
    assert obj.param0 == 1
    assert obj.param1 == 1
    assert obj.param2 is None


def test_customizable_missing_required_param():
    config = {'param2': 'test'}
    with pytest.raises(ValidationError):
        MyCustomizable.from_config(config)


def test_customizable_invalid_param_type():
    config = {'param1': 'not an int', 'param2': 'test'}
    with pytest.raises(ValidationError):
        MyCustomizable.from_config(config)


# Test cases for the TypedCustomizable class
class BaseAlgorithm(TypedCustomizable):
    config_schema = TypedCustomizable.config_schema.copy()
    aliases = ['base_algorithm']


class AlgorithmA(BaseAlgorithm):
    aliases = ['algorithm_a']
    config_schema = {
        'type': Schema(str),
        'param_a': Schema(int, default=5),
    }

    def __init__(self, param_a):
        self.param_a = param_a


class AlgorithmB(BaseAlgorithm):
    aliases = ['algorithm_b']
    config_schema = {
        'type': Schema(str),
        'param_b': Schema(str),
    }

    def __init__(self, param_b):
        self.param_b = param_b


def test_typed_customizable_from_config_algorithm_a():
    config = {'type': 'algorithm_a', 'param_a': 10}
    obj = BaseAlgorithm.from_config(config)
    assert isinstance(obj, AlgorithmA)
    assert obj.param_a == 10


def test_typed_customizable_from_config_algorithm_b():
    config = {'type': 'algorithm_b', 'param_b': 'value'}
    obj = BaseAlgorithm.from_config(config)
    assert isinstance(obj, AlgorithmB)
    assert obj.param_b == 'value'


def test_typed_customizable_missing_type():
    config = {'param_a': 10}
    with pytest.raises(ValueError):
        BaseAlgorithm.from_config(config)


def test_typed_customizable_invalid_type():
    config = {'type': 'unknown_type', 'param_a': 10}
    with pytest.raises(Exception):
        BaseAlgorithm.from_config(config)


def test_typed_customizable_missing_required_param():
    config = {'type': 'algorithm_b'}
    with pytest.raises(ValidationError):
        BaseAlgorithm.from_config(config)
