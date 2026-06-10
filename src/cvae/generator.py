from __future__ import annotations

from typing import Any

import torch

from cvae.cvae import DecoderBase
from cvae.schema import ConditionSchema


class ConditionedGenerator:
    def __init__(
        self,
        decoder: DecoderBase,
        schema: ConditionSchema,
        seed: int,
    ) -> None:
        self.decoder = decoder
        self.latent_dim = decoder.latent_dim
        self.schema = schema
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    @torch.no_grad()
    def generate(self, values: dict[str, Any]) -> torch.Tensor:
        """Sample one example from the decoder conditioned on *values*.

        Each call advances the internal RNG so successive calls produce
        different samples even for identical condition values.
        """
        try:
            device = next(self.decoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        z = torch.randn(1, self.latent_dim, generator=self._rng, device=device)
        condition = self.schema.encode(values).unsqueeze(0).to(device)

        self.decoder.eval()
        output = self.decoder(z, condition)
        return output.squeeze(0)