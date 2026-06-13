import enum
import torch
import pytest
from cvae import ConditionSchema, ConditionVariable
from cvae import InputEncoderCategoricalToOneHot, InputEncoderNormalizedRange, InputEncoderBoolean


def build_test_schema():
    schema = ConditionSchema([
        ConditionVariable(name='category', encoder=InputEncoderCategoricalToOneHot(num_classes=5)),
        ConditionVariable(name='range', encoder=InputEncoderNormalizedRange(in_min=0.0, in_max=10.0, out_min=-1.0, out_max=1.0)),
        ConditionVariable(name='bool', encoder=InputEncoderBoolean(true_value=5.0, false_value=-5.0))]
    )
    return schema


def build_invalid_test_schema():
    schema = ConditionSchema([
        ConditionVariable(name='var1', encoder=InputEncoderCategoricalToOneHot(num_classes=5)),
        ConditionVariable(name='var1', encoder=InputEncoderNormalizedRange(in_min=0.0, in_max=10.0, out_min=-1.0, out_max=1.0)),
        ConditionVariable(name='var2', encoder=InputEncoderBoolean(true_value=5.0, false_value=-5.0))]
    )
    return schema


def build_test_values():
    values = {
        'category': 0,
        'range': 0.0,
        'bool': True
    }
    return values

epsilon = 0.0001


def test_condition_schema_output_dims():
    schema = build_test_schema()
    assert schema.output_dim == 5 + 1 + 1


def test_condition_schema_variable_names():
    schema = build_test_schema()
    assert schema.variable_names == ['category', 'range', 'bool']


def test_condition_schema_variables():
    schema = build_test_schema()
    assert len(schema.variables) == 3
    assert all([isinstance(x, ConditionVariable) for x in schema.variables])


def test_condition_schema_encode():
    schema = build_test_schema()
    values = build_test_values()
    tensor = schema.encode(values)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.numel() == 5 + 1 + 1
    assert tensor.dim() == 1


def test_condition_schema_split():
    schema = build_test_schema()
    values = build_test_values()
    tensor = schema.encode(values)
    split = schema.split(tensor)
    assert isinstance(split, dict)
    assert len(split) == 3
    assert split['category'][0] == 1
    assert abs(split['range'] - -1.0) < epsilon
    assert abs(split['bool'] - 5.0) < epsilon


def test_condition_scheme_duplicate_names():
    with pytest.raises(ValueError):
        build_invalid_test_schema()
