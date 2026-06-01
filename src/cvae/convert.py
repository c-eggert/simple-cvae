import abc
import enum

from typing import Generic, TypeVar

import torch
from torch import Tensor

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class InputEncoder(abc.ABC, Generic[InputT, OutputT]):
    @abc.abstractmethod
    def encode(self, value: InputT) -> OutputT:
        raise NotImplementedError(
            '"encode" method has not been implemented for this type.'
        )

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError(
            '"output_dim" method has not been implemented for this type.'
        )

    def __call__(self, value: InputT) -> OutputT:
        return self.encode(value)


class InputEncoderCategoricalToOneHot(InputEncoder[int | enum.Enum, Tensor]):
    def __init__(self, num_classes: int, dtype=torch.float32) -> None:
        self._num_classes = num_classes
        self._dtype = dtype

    def encode(self, value: int | enum.Enum) -> OutputT:
        index = self._resolve_index(value)
        one_hot = torch.zeros(self._num_classes, dtype=self._dtype)
        one_hot[index] = 1
        return one_hot

    @property
    def output_dim(self) -> int:
        return self._num_classes

    def _resolve_index(self, value: int | enum.Enum) -> int:
        if isinstance(value, int):
            index = value
        elif isinstance(value, enum.Enum):
            index = value.value
        else:
            raise TypeError(f"Expected int or enum, got {type(value)}")

        if not (0 <= index < self._num_classes):
            raise ValueError(f"Index {index} is out of range [0, {self._num_classes})")
        return index


class InputEncoderNormalizedRange(InputEncoder[float, Tensor]):
    def __init__(
        self,
        in_min: float,
        in_max: float,
        out_min: float,
        out_max: float,
        dtype=torch.float32,
    ) -> None:
        self._in_min = in_min
        self._in_max = in_max
        self._out_min = out_min
        self._out_max = out_max
        self._dtype = dtype

    def encode(self, value: InputT) -> OutputT:
        if not (self._in_min <= value <= self._in_max):
            raise ValueError(
                f"Value ({value}) is out of range ({self._in_min}, {self._in_max})"
            )

        value_norm = (value - self._in_min) / (self._in_max - self._in_min)
        value_norm = value_norm * (self._out_max - self._out_min) + self._out_min
        return torch.tensor([value_norm], dtype=self._dtype)

    @property
    def output_dim(self) -> int:
        return 1
