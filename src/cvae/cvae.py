from __future__ import annotations

import abc
from pathlib import Path
from typing import Type, Tuple

import torch
import torch.nn as nn
from torch import nn as nn

DEFAULT_FILM_TARGETS: tuple[Type[nn.Module], ...] = (
    nn.Conv2d,
    nn.ConvTranspose2d,
)

DEFAULT_OUTPUT_FEATURE_ATTRS: tuple[str, ...] = (
    "out_features",
    "out_channels",
)


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, num_channels: int):
        """

        :param num_channels: Number of channels over which FiLM is applied
        :param cond_dim: Number of dimensions of conditioning
        """
        super(FiLMLayer, self).__init__()
        self.modulator = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, h, c):
        # h: [B, C, H, W]
        # c: [B, cond_dim]
        gamma, beta = self.modulator(c).chunk(2, dim=-1)

        # reshape for broadcasting over H, W
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        return gamma * h + beta


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels_data: int,
        in_channels_cond: int,
        out_channels: int,
        num_groups: int,
        stride: int,
        padding: int,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels_data, out_channels, 3, stride, padding=padding
        )
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels, affine=False
        )
        self.film = FiLMLayer(cond_dim=in_channels_cond, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, data: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.conv(data)
        x = self.norm(x)
        x = self.film(x, condition)
        x = self.act(x)
        return x


class EncoderBase(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self, data: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode data into (mu, logvar) given a condition."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels_data: int,
        in_channels_cond: int,
        out_channels: int,
        num_groups: int,
        stride: int,
        padding: int,
    ):
        super(UpConvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(
            in_channels=in_channels_data,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(
            num_groups=num_groups, num_channels=out_channels, affine=False
        )
        self.film = FiLMLayer(cond_dim=in_channels_cond, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, data: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.up(data)
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, condition)
        x = self.act(x)
        return x


class DecoderBase(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, sampled: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode a latent sample into a reconstruction given a condition."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        state_dict = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(state_dict)


class ConditionalGenerativeModel(nn.Module, abc.ABC):
    """
    Interface that every model passed to Trainer must satisfy.

    forward() receives
      - sample    : the tensor to be reconstructed   (B, *)
      - condition : the conditioning tensor          (B, *)

    forward() returns
      - reconstruction : tensor with the same shape as `sample`
      - latent_params  : tuple (mu, logvar), each of shape (B, latent_dim)
    """

    @abc.abstractmethod
    def forward(
        self, sample: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError('Abstract "forward" method was not implemented.')


class CVAE(ConditionalGenerativeModel):
    def __init__(self, encoder: EncoderBase, decoder: DecoderBase):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, sample: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(sample, condition)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        reconst = self.decoder(z, condition)
        reconst = reconst[tuple(slice(s) for s in sample.shape)]
        return reconst, (mu, logvar)

    def save_encoder(self, path: str | Path) -> None:
        self.encoder.save(path)

    def load_encoder(self, path: str | Path) -> None:
        self.encoder.load(path)

    def save_decoder(self, path: str | Path) -> None:
        self.decoder.save(path)

    def load_decoder(self, path: str | Path) -> None:
        self.decoder.load(path)


class FiLMConditioningInjector(object):
    def __init__(
        self,
        cond_dim: int,
        target_layer_types: tuple[Type[nn.Module], ...] = DEFAULT_FILM_TARGETS,
        output_dim_attrs: tuple[str, ...] = DEFAULT_OUTPUT_FEATURE_ATTRS,
    ):
        self._cond_dim = cond_dim
        self._target_layer_types = target_layer_types
        self._output_dim_attrs = output_dim_attrs

    def inject(self, module: nn.Module) -> Tuple[nn.Module, int, int]:
        num_inserted_film = 0
        num_inserted_norm = 0
        # recursively traverse the module to find nn.Sequential
        for name, child in list(module.named_children()):
            rebuilt_child, n_film, n_norm = self.inject(child)
            if rebuilt_child is not child:
                setattr(module, name, rebuilt_child)
            num_inserted_film += n
            num_inserted_norm += n_film

        # rebuild all nn.Sequential containers with inserted FiLM layers
        if isinstance(module, nn.Sequential):
            new_layers = list[nn.Module] = []
            for layer in module.children():
                new_layers.append(layer)
                if isinstance(layer, self._target_layer_types):
                    layer_out_dim = self._infer_output_dim(layer)
                    film_layer = FiLMLayer(
                        cond_dim=self._cond_dim, num_channels=layer_out_dim
                    )
                    new_layers.append(film_layer)
                    num_inserted_film += 1

    def _infer_output_dim(self, layer) -> int:
        for attr in self._output_dim_attrs:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(
            f"Cannot infer output dimensions for layer {layer.name} (Type: {type(layer).__name__})"
        )


class FiLMConditionedEncoder(nn.Module):
    def __init__(self, encoder: nn.Module):
        super(FiLMConditionedEncoder, self).__init__()

    def forward(self, x):
        pass

    def _inject_film_layers(self, encoder):
        num_inserted = 0
        for name, child in list(encoder.named_children()):
            child_rebuilt, n = self._inject_film_layers(child)
            if child_rebuilt is not child:
                pass
