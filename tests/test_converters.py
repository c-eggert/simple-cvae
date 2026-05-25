import enum
import torch
import pytest
from cvae import InputEncoderCategoricalToOneHot, InputEncoderNormalizedRange


class EnumTestValsContinuous(enum.Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


def test_int_to_one_hot_min():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    enc_val = encoder.encode(0)
    assert enc_val[0] == 1
    assert encoder.output_dim() == 5

def test_int_to_one_hot_max():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    enc_val = encoder.encode(4)
    assert enc_val[4] == 1
    assert encoder.output_dim() == 5

def test_int_to_one_hot_max_unknown_data():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    with pytest.raises(TypeError):
        encoder.encode('test')

def test_int_to_one_hot_max_wrong_range_low():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    with pytest.raises(ValueError):
        encoder.encode(-1)

def test_int_to_one_hot_max_wrong_range_high():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    with pytest.raises(ValueError):
        encoder.encode(5)

def test_enum_to_one_hot_min():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    enc_val = encoder.encode(EnumTestValsContinuous.ZERO)
    assert enc_val[0] == 1
    assert encoder.output_dim() == 5

def test_enum_to_one_hot_max():
    encoder = InputEncoderCategoricalToOneHot(num_classes=5, dtype=torch.int32)
    enc_val = encoder.encode(EnumTestValsContinuous.FOUR)
    assert enc_val[4] == 1
    assert encoder.output_dim() == 5

def test_range_normalize_min():
    encoder = InputEncoderNormalizedRange(in_min=10.0, in_max=20.0, out_min=-1.0, out_max=1.0)
    enc_val = encoder.encode(10.0)
    assert abs(enc_val.numpy() + 1.0) < 0.0001

def test_range_normalize_max():
    encoder = InputEncoderNormalizedRange(in_min=10.0, in_max=20.0, out_min=-1.0, out_max=1.0)
    enc_val = encoder.encode(20.0)
    assert abs(enc_val.numpy() - 1.0) < 0.0001

def test_range_normalize_wrong_range_low():
    encoder = InputEncoderNormalizedRange(in_min=10.0, in_max=20.0, out_min=-1.0, out_max=1.0)
    with pytest.raises(ValueError):
        encoder.encode(9.0)

def test_range_normalize_wrong_range_high():
    encoder = InputEncoderNormalizedRange(in_min=10.0, in_max=20.0, out_min=-1.0, out_max=1.0)
    with pytest.raises(ValueError):
        encoder.encode(21.0)
