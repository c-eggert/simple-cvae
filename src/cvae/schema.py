from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cvae.convert import InputEncoder


@dataclass
class ConditionVariable:
    """Pairs a variable name with its encoder."""

    name: str
    encoder: InputEncoder

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim


class ConditionSchema:
    """
    Describes the full set of conditional variables and encodes a dict of
    raw values into a single flat condition tensor.

    Usage::

        schema = ConditionSchema([
            ConditionVariable("style",    InputEncoderCategoricalToOneHot(num_classes=4)),
            ConditionVariable("severity", InputEncoderCategoricalToOneHot(num_classes=3)),
            ConditionVariable("ratio",    InputEncoderNormalizedRange(0.0, 1.0, -1.0, 1.0)),
        ])

        # single sample — use this inside a Dataset.__getitem__
        cond = schema.encode({"style": 2, "severity": 0, "ratio": 0.75})
        # cond.shape == (8,)  [4 + 3 + 1]

        print(schema.output_dim)  # 8
    """

    def __init__(self, variables: list[ConditionVariable]) -> None:
        names = [v.name for v in variables]
        if len(set(names)) != len(names):
            raise ValueError("Variable names in a ConditionSchema must be unique.")
        self._variables = list(variables)

    @property
    def output_dim(self) -> int:
        """Total length of the encoded condition vector."""
        return sum(v.output_dim for v in self._variables)

    @property
    def variables(self) -> list[ConditionVariable]:
        return list(self._variables)

    @property
    def variable_names(self) -> list[str]:
        return [v.name for v in self._variables]

    def encode(self, values: dict[str, Any]) -> torch.Tensor:
        """Encode one sample.  Missing keys raise ``KeyError``."""
        parts = [v.encoder.encode(values[v.name]) for v in self._variables]
        return torch.cat(parts, dim=0)
