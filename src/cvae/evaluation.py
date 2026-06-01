from __future__ import annotations

import itertools
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cvae.convert import InputEncoderCategoricalToOneHot, InputEncoderNormalizedRange
from cvae.cvae import CVAE, EncoderBase
from cvae.schema import ConditionSchema, ConditionVariable


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_condition_grid(
    schema: ConditionSchema, num_ranged_steps: int
) -> list[dict]:
    """Cartesian product of per-variable candidate values.

    Categorical variables: all class indices.
    Ranged variables: ``num_ranged_steps`` evenly spaced values in [in_min, in_max].
    """
    per_variable: list[Sequence] = []
    for v in schema.variables:
        if isinstance(v.encoder, InputEncoderCategoricalToOneHot):
            per_variable.append(range(v.encoder.num_classes))
        elif isinstance(v.encoder, InputEncoderNormalizedRange):
            lo, hi = v.encoder.in_min, v.encoder.in_max
            per_variable.append(
                [lo + i * (hi - lo) / (num_ranged_steps - 1) for i in range(num_ranged_steps)]
            )
        else:
            raise TypeError(f"Unsupported encoder type for grid: {type(v.encoder)}")

    return [
        dict(zip(schema.variable_names, combo))
        for combo in itertools.product(*per_variable)
    ]


# ---------------------------------------------------------------------------
# Condition predictor
# ---------------------------------------------------------------------------

class ConditionPredictor(nn.Module):
    """Predicts conditional variables from a sample.

    Uses the given EncoderBase as a backbone, but passes a **zero condition
    vector** so that FiLM layers receive no actual conditioning signal.  A
    separate linear head per variable maps the latent mean ``mu`` to either
    class logits (categorical) or a scalar estimate (ranged).

    Parameters
    ----------
    encoder:
        An EncoderBase instance to use as the feature extractor.  Pass
        ``copy.deepcopy(cvae.encoder)`` to initialise from trained weights
        without modifying the original CVAE.
    schema:
        The condition schema that describes all conditional variables.
    latent_dim:
        Dimensionality of the latent space (= the size of ``mu`` produced by
        the encoder).
    """

    def __init__(
        self,
        encoder: EncoderBase,
        schema: ConditionSchema,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.schema = schema
        self.latent_dim = latent_dim
        self.heads = nn.ModuleDict(
            {v.name: nn.Linear(latent_dim, v.output_dim) for v in schema.variables}
        )

    def forward(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        B = data.shape[0]
        zero_cond = torch.zeros(B, self.schema.output_dim, device=data.device)
        mu, _ = self.encoder(data, zero_cond)
        return {name: head(mu) for name, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Evaluation 1 — MSE by category
# ---------------------------------------------------------------------------

class MSEByCategoryEvaluation:
    """Measures reconstruction MSE broken down by the value of each categorical
    variable in the schema.

    Results are written to ``<output_dir>/mse_by_category.txt``.
    """

    def __init__(
        self,
        model: CVAE,
        schema: ConditionSchema,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.device = torch.device(device) if device else _default_device()
        self.model = model.to(self.device).eval()
        self.schema = schema

    @torch.no_grad()
    def run(
        self,
        dataset: Dataset,
        output_dir: str | Path,
        batch_size: int = 64,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        categorical = [
            v for v in self.schema.variables
            if isinstance(v.encoder, InputEncoderCategoricalToOneHot)
        ]
        if not categorical:
            print("No categorical variables in schema; skipping MSE-by-category.")
            return

        # {var_name: {class_idx: [per-sample MSE]}}
        accum: dict[str, dict[int, list[float]]] = {
            v.name: {c: [] for c in range(v.encoder.num_classes)}
            for v in categorical
        }

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for samples, conditions in loader:
            samples = samples.to(self.device)
            conditions = conditions.to(self.device)

            reconstructions, _ = self.model(samples, conditions)
            # per-sample mean MSE over all spatial / channel dims
            sample_mse = (
                nn.functional.mse_loss(reconstructions, samples, reduction="none")
                .flatten(1)
                .mean(1)
            )  # (B,)

            slices = self.schema.split(conditions)
            for v in categorical:
                class_idx = slices[v.name].argmax(dim=1)
                for cls, mse in zip(class_idx.tolist(), sample_mse.tolist()):
                    accum[v.name][cls].append(mse)

        report_path = output_dir / "mse_by_category.txt"
        with open(report_path, "w") as f:
            for v in categorical:
                f.write(f"Variable: {v.name}\n")
                f.write(f"{'Class':>8}  {'N':>8}  {'Mean MSE':>12}  {'Std MSE':>12}\n")
                f.write("-" * 48 + "\n")
                for cls in range(v.encoder.num_classes):
                    vals = accum[v.name][cls]
                    n = len(vals)
                    if n:
                        mean = sum(vals) / n
                        std = (sum((x - mean) ** 2 for x in vals) / n) ** 0.5
                    else:
                        mean = std = float("nan")
                    f.write(f"{cls:>8}  {n:>8}  {mean:>12.6f}  {std:>12.6f}\n")
                f.write("\n")

        print(f"MSE-by-category report written to {report_path}")


# ---------------------------------------------------------------------------
# Evaluation 2 — conditional prediction accuracy
# ---------------------------------------------------------------------------

class ConditionalPredictionEvaluation:
    """Trains a :class:`ConditionPredictor` on real samples, then generates
    samples from the CVAE across the full conditional grid and measures how
    well the predictor can recover the intended conditionals.

    Categorical variables are evaluated with **accuracy**.
    Ranged variables are evaluated with **mean absolute error (MAE)**.

    Results are written to ``<output_dir>/conditional_prediction.txt``.

    Parameters
    ----------
    model:
        The trained CVAE.
    schema:
        Condition schema used during training.
    predictor_encoder:
        An EncoderBase to use as the predictor backbone.  Pass
        ``copy.deepcopy(model.encoder)`` to start from the CVAE's learned
        weights, or a fresh instance to train from scratch.
    latent_dim:
        Size of the latent vector ``z`` (must match what the CVAE was trained
        with).
    device:
        Compute device.  Auto-detected when ``None``.
    num_ranged_steps:
        Number of evenly-spaced evaluation points for ranged variables.
    num_samples_per_condition:
        How many independent samples to generate per condition point.
    generation_seed:
        RNG seed used when drawing latent samples for generation, ensuring
        reproducible outputs.
    """

    def __init__(
        self,
        model: CVAE,
        schema: ConditionSchema,
        predictor_encoder: EncoderBase,
        latent_dim: int,
        device: Optional[str | torch.device] = None,
        num_ranged_steps: int = 10,
        num_samples_per_condition: int = 16,
        generation_seed: int = 42,
    ) -> None:
        self.device = torch.device(device) if device else _default_device()
        self.model = model.to(self.device).eval()
        self.schema = schema
        self.latent_dim = latent_dim
        self.predictor = ConditionPredictor(
            predictor_encoder, schema, latent_dim
        ).to(self.device)
        self.num_ranged_steps = num_ranged_steps
        self.num_samples_per_condition = num_samples_per_condition
        self.generation_seed = generation_seed

    def run(
        self,
        dataset: Dataset,
        output_dir: str | Path,
        predictor_epochs: int = 20,
        predictor_lr: float = 1e-3,
        batch_size: int = 64,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._train_predictor(dataset, predictor_epochs, predictor_lr, batch_size)
        self._evaluate_on_generated(output_dir)

    # ------------------------------------------------------------------
    # predictor training
    # ------------------------------------------------------------------

    def _train_predictor(
        self,
        dataset: Dataset,
        epochs: int,
        lr: float,
        batch_size: int,
    ) -> None:
        print("Training condition predictor on real samples...")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        log_every = max(1, epochs // 5)

        self.predictor.train()
        for epoch in range(1, epochs + 1):
            total = 0.0
            for samples, conditions in loader:
                samples = samples.to(self.device)
                conditions = conditions.to(self.device)
                loss = self._predictor_loss(self.predictor(samples), conditions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            if epoch % log_every == 0:
                print(f"  epoch {epoch:>3}/{epochs}  loss={total / len(loader):.4f}")

        self.predictor.eval()
        print("Predictor training complete.")

    def _predictor_loss(
        self,
        predictions: dict[str, torch.Tensor],
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=self.device).squeeze()
        slices = self.schema.split(conditions)
        for v in self.schema.variables:
            pred = predictions[v.name]
            cond = slices[v.name]
            if isinstance(v.encoder, InputEncoderCategoricalToOneHot):
                loss = loss + nn.functional.cross_entropy(pred, cond.argmax(dim=1))
            elif isinstance(v.encoder, InputEncoderNormalizedRange):
                loss = loss + nn.functional.mse_loss(pred, cond)
        return loss

    # ------------------------------------------------------------------
    # generation + evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_on_generated(self, output_dir: Path) -> None:
        condition_grid = _build_condition_grid(self.schema, self.num_ranged_steps)

        rng = torch.Generator(device=self.device)
        rng.manual_seed(self.generation_seed)

        results: dict[str, list[dict]] = {v.name: [] for v in self.schema.variables}

        for cond_values in condition_grid:
            cond_tensor = self.schema.encode(cond_values).to(self.device)
            # (num_samples, cond_dim)
            cond_batch = cond_tensor.unsqueeze(0).expand(self.num_samples_per_condition, -1)

            z = torch.randn(
                self.num_samples_per_condition,
                self.latent_dim,
                device=self.device,
                generator=rng,
            )
            generated = self.model.decoder(z, cond_batch)
            predictions = self.predictor(generated)

            slices = self.schema.split(cond_batch)
            for v in self.schema.variables:
                pred = predictions[v.name]
                true = slices[v.name]
                if isinstance(v.encoder, InputEncoderCategoricalToOneHot):
                    acc = (pred.argmax(1) == true.argmax(1)).float().mean().item()
                    results[v.name].append({"condition": cond_values[v.name], "accuracy": acc})
                elif isinstance(v.encoder, InputEncoderNormalizedRange):
                    mae = (pred - true).abs().mean().item()
                    results[v.name].append({"condition": cond_values[v.name], "mae": mae})

        self._write_report(results, output_dir)

    def _write_report(
        self, results: dict[str, list[dict]], output_dir: Path
    ) -> None:
        n = self.num_samples_per_condition
        report_path = output_dir / "conditional_prediction.txt"
        with open(report_path, "w") as f:
            for v in self.schema.variables:
                rows = results[v.name]
                f.write(f"Variable: {v.name}  (n={n} generated samples per condition)\n")
                if isinstance(v.encoder, InputEncoderCategoricalToOneHot):
                    f.write(f"{'Class':>8}  {'Accuracy':>10}\n")
                    f.write("-" * 24 + "\n")
                    for r in rows:
                        f.write(f"{r['condition']:>8}  {r['accuracy']:>10.4f}\n")
                    mean_acc = sum(r["accuracy"] for r in rows) / len(rows)
                    f.write(f"{'mean':>8}  {mean_acc:>10.4f}\n")
                elif isinstance(v.encoder, InputEncoderNormalizedRange):
                    f.write(f"{'Value':>10}  {'MAE':>10}\n")
                    f.write("-" * 26 + "\n")
                    for r in rows:
                        f.write(f"{r['condition']:>10.4f}  {r['mae']:>10.4f}\n")
                    mean_mae = sum(r["mae"] for r in rows) / len(rows)
                    f.write(f"{'mean':>10}  {mean_mae:>10.4f}\n")
                f.write("\n")

        print(f"Conditional prediction report written to {report_path}")