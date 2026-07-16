import inspect

import torch

import lis2.utils.schema as schema


def test_get_primary_type():
    assert schema.get_primary_type(float | torch.Tensor) == float
    assert schema.get_primary_type(float | int) == int
    assert schema.get_primary_type(str | None) == str
    assert schema.get_primary_type(bool | None) == bool
    assert schema.get_primary_type(int | None) == int
    assert schema.get_primary_type(torch.FloatTensor) == torch.FloatTensor


def test_infer_type_from_annotation():
    init_signature = inspect.signature(torch.optim.Adam.__init__)
    res1 = schema.infer_type_from_annotation(init_signature.parameters["lr"].annotation)
    assert res1 == float
    res2 = schema.infer_type_from_annotation(
        init_signature.parameters["betas"].annotation
    )
    assert res2 == (float, float)


def test_infer_type_from_default():
    init_signature = inspect.signature(torch.optim.Adam.__init__)
    res1 = schema.infer_type_from_default(init_signature.parameters["lr"].default)
    assert res1 == float
    res2 = schema.infer_type_from_default(init_signature.parameters["betas"].default)
    assert res2 == (float, float)


def test_generate_config_schema():
    schema_adam = schema.generate_config_schema(torch.optim.Adam)
    assert len(schema_adam) == 11
    assert schema_adam["lr"].default == 0.001
