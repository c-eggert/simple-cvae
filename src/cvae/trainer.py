from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


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


# (reconstruction, sample) -> scalar
ReconLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# (mu, logvar) -> scalar
LatentLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mse_recon_loss(reconstruction: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(reconstruction, sample, reduction="mean")


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL( N(mu, exp(logvar)) || N(0,1) ), averaged over batch and latent dims
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


class Trainer:
    def __init__(
        self,
        model: ConditionalGenerativeModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        recon_loss_fn: Optional[ReconLossFn] = None,
        latent_loss_fn: Optional[LatentLossFn] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[str | torch.device] = None,
        kl_weight: float = 1.0,
    ) -> None:
        """
        Args:
            model:           A model satisfying ConditionalGenerativeModel.
            train_loader:    DataLoader that yields (sample, condition) batches.
            val_loader:      Optional DataLoader for validation (same format).
            recon_loss_fn:   ``fn(reconstruction, sample) -> scalar``.
                             Defaults to MSE.
            latent_loss_fn:  ``fn(mu, logvar) -> scalar``.
                             Defaults to standard KL divergence.
            optimizer:       Any torch Optimizer.  Defaults to Adam(lr=1e-3).
            device:          'cpu', 'cuda', 'mps', or a torch.device.
                             Auto-detected when None.
            kl_weight:       Weight applied to the latent loss term.
        """
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.recon_loss_fn: ReconLossFn = recon_loss_fn or mse_recon_loss
        self.latent_loss_fn: LatentLossFn = latent_loss_fn or kl_divergence_loss
        self.kl_weight = kl_weight

        self.optimizer: Optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(self.model.parameters(), lr=1e-3)
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_recon_loss": [],
            "val_recon_loss": [],
            "train_kl_loss": [],
            "val_kl_loss": [],
        }

    def _step(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass + loss computation. Returns (total, recon, latent)."""
        sample, condition = batch
        sample = sample.to(self.device)
        condition = condition.to(self.device)

        reconstruction, (mu, logvar) = self.model(sample, condition)
        recon_loss = self.recon_loss_fn(reconstruction, sample)
        latent_loss = self.latent_loss_fn(mu, logvar)
        total_loss = recon_loss + self.kl_weight * latent_loss
        return total_loss, recon_loss, latent_loss

    def _train_epoch(self) -> Tuple[float, float, float]:
        self.model.train()
        total, recon, kl = 0.0, 0.0, 0.0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss, recon_loss, latent_loss = self._step(batch)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            recon += recon_loss.item()
            kl += latent_loss.item()

        n = len(self.train_loader)
        return total / n, recon / n, kl / n

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, float, float]:
        self.model.eval()
        total, recon, kl = 0.0, 0.0, 0.0

        for batch in self.val_loader:
            loss, recon_loss, latent_loss = self._step(batch)
            total += loss.item()
            recon += recon_loss.item()
            kl += latent_loss.item()

        n = len(self.val_loader)
        return total / n, recon / n, kl / n

    def fit(
        self,
        epochs: int,
        print_every: int = 1,
    ) -> None:
        for epoch in range(1, epochs + 1):
            train_loss, train_recon, train_kl = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_recon_loss"].append(train_recon)
            self.history["train_kl_loss"].append(train_kl)

            val_loss: Optional[float] = None
            if self.val_loader is not None:
                val_loss, val_recon, val_kl = self._val_epoch()
                self.history["val_loss"].append(val_loss)
                self.history["val_recon_loss"].append(val_recon)
                self.history["val_kl_loss"].append(val_kl)

            if print_every and epoch % print_every == 0:
                msg = (
                    f"Epoch {epoch:>4}/{epochs}"
                    f"  train_loss={train_loss:.6f}"
                    f"  (recon={train_recon:.6f}, kl={train_kl:.6f})"
                )
                if val_loss is not None:
                    msg += (
                        f"  val_loss={val_loss:.6f}"
                        f"  (recon={val_recon:.6f}, kl={val_kl:.6f})"
                    )
                print(msg)

            checkpoint_path = os.path.join(
                "checkpoints",
                f"epoch_{epoch:0>4}_loss{'' if val_loss is None else val_loss}.pth",
            )
            self.save(checkpoint_path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
