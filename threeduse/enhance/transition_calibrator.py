"""Paired-data calibration of UIE appearance-transition proposals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .effective_transition import (
    GLOBAL_OPERATOR_DIM,
    GLOBAL_TRANSITION_CONDITION_DIM,
    LOCAL_CONDITION_DIM,
    LOCAL_OPERATOR_DIM,
    LOCAL_TRANSITION_CONDITION_DIM,
    GLOBAL_CONDITION_DIM,
)


SCHEMA_NAME = "3duse_transition_calibrator"
SCHEMA_VERSION = 5


def _regressor(input_dim: int, output_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class DeterministicTransitionCalibrator(nn.Module):
    """Map a frozen UIE proposal into the paired transition domain.

    The two residual MLPs predict corrections to the proposal's global affine
    operator and local diagonal gain/bias operators. They are trained only on
    paired 2D data and are frozen while optimizing a target 3D scene.
    """

    def __init__(
        self,
        *,
        global_hidden_dim: int = 128,
        local_hidden_dim: int = 256,
        condition_mode: str = "source_and_proposal",
        prediction_mode: str = "proposal_residual",
    ) -> None:
        super().__init__()
        if global_hidden_dim < 8 or local_hidden_dim < 8:
            raise ValueError("Calibrator hidden dimensions must be at least eight")
        if condition_mode != "source_and_proposal":
            raise ValueError(
                "The released calibrator requires source_and_proposal conditioning"
            )
        if prediction_mode != "proposal_residual":
            raise ValueError(
                "The released calibrator predicts residuals over UIE proposals"
            )
        self.global_hidden_dim = int(global_hidden_dim)
        self.local_hidden_dim = int(local_hidden_dim)
        self.condition_mode = "source_and_proposal"
        self.target_mode = "mlp"
        self.prediction_mode = "proposal_residual"

        self.global_head = _regressor(
            GLOBAL_TRANSITION_CONDITION_DIM,
            GLOBAL_OPERATOR_DIM,
            self.global_hidden_dim,
        )
        self.local_head = _regressor(
            LOCAL_TRANSITION_CONDITION_DIM,
            LOCAL_OPERATOR_DIM,
            self.local_hidden_dim,
        )
        self.register_buffer("global_median", torch.zeros(GLOBAL_OPERATOR_DIM))
        self.register_buffer("global_scale", torch.ones(GLOBAL_OPERATOR_DIM))
        self.register_buffer("local_median", torch.zeros(LOCAL_OPERATOR_DIM))
        self.register_buffer("local_scale", torch.ones(LOCAL_OPERATOR_DIM))
        self.register_buffer(
            "global_condition_median", torch.zeros(GLOBAL_TRANSITION_CONDITION_DIM)
        )
        self.register_buffer(
            "global_condition_scale", torch.ones(GLOBAL_TRANSITION_CONDITION_DIM)
        )
        self.register_buffer(
            "local_condition_median", torch.zeros(LOCAL_TRANSITION_CONDITION_DIM)
        )
        self.register_buffer(
            "local_condition_scale", torch.ones(LOCAL_TRANSITION_CONDITION_DIM)
        )

    @staticmethod
    def _normalize(value: Tensor, median: Tensor, scale: Tensor) -> Tensor:
        return (value - median.to(value)) / scale.to(value).clamp_min(1e-5)

    @staticmethod
    def _denormalize(value: Tensor, median: Tensor, scale: Tensor) -> Tensor:
        return value * scale.to(value) + median.to(value)

    @torch.no_grad()
    def set_normalization(
        self,
        *,
        global_operator: Tensor,
        local_operator: Tensor,
        global_condition: Tensor,
        local_condition: Tensor,
    ) -> None:
        entries = (
            (global_operator, self.global_median, self.global_scale),
            (local_operator, self.local_median, self.local_scale),
            (
                global_condition,
                self.global_condition_median,
                self.global_condition_scale,
            ),
            (
                local_condition,
                self.local_condition_median,
                self.local_condition_scale,
            ),
        )
        for samples, median_buffer, scale_buffer in entries:
            flat = samples.reshape(-1, samples.shape[-1]).float()
            median = flat.median(dim=0).values
            q25 = torch.quantile(flat, 0.25, dim=0)
            q75 = torch.quantile(flat, 0.75, dim=0)
            median_buffer.copy_(median.to(median_buffer))
            scale_buffer.copy_((q75 - q25).clamp_min(1e-3).to(scale_buffer))

    def normalize_global_operator(self, operator: Tensor) -> Tensor:
        return self._normalize(operator, self.global_median, self.global_scale)

    def normalize_local_operator(self, operator: Tensor) -> Tensor:
        return self._normalize(operator, self.local_median, self.local_scale)

    def global_target(self, condition: Tensor) -> Tensor:
        condition_n = self._normalize(
            condition,
            self.global_condition_median,
            self.global_condition_scale,
        )
        residual_n = self.global_head(condition_n)
        residual = self._denormalize(
            residual_n,
            self.global_median,
            self.global_scale,
        )
        proposal = condition[..., -GLOBAL_OPERATOR_DIM:]
        return proposal + residual

    def local_target(self, condition: Tensor) -> Tensor:
        condition_n = self._normalize(
            condition,
            self.local_condition_median,
            self.local_condition_scale,
        )
        residual_n = self.local_head(condition_n)
        residual = self._denormalize(
            residual_n,
            self.local_median,
            self.local_scale,
        )
        proposal_start = (
            LOCAL_CONDITION_DIM + GLOBAL_CONDITION_DIM + GLOBAL_OPERATOR_DIM
        )
        proposal = condition[
            ..., proposal_start : proposal_start + LOCAL_OPERATOR_DIM
        ]
        return proposal + residual

    def checkpoint_payload(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "model_config": {
                "global_hidden_dim": self.global_hidden_dim,
                "local_hidden_dim": self.local_hidden_dim,
                "condition_mode": self.condition_mode,
                "target_mode": self.target_mode,
                "prediction_mode": self.prediction_mode,
            },
            "state_dict": self.state_dict(),
            "metadata": dict(metadata),
        }

    @classmethod
    def from_checkpoint(
        cls, path: str | Path
    ) -> tuple["DeterministicTransitionCalibrator", dict[str, Any]]:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected a mapping checkpoint at {path}")
        schema = payload.get("schema")
        version = int(payload.get("schema_version", -1))
        if schema != SCHEMA_NAME or version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported calibrator {schema!r} version {version!r}"
            )

        config = dict(payload.get("model_config", {}))
        expected = {
            "condition_mode": "source_and_proposal",
            "target_mode": "mlp",
            "prediction_mode": "proposal_residual",
        }
        for name, value in expected.items():
            if config.get(name, value) != value:
                raise ValueError(
                    f"Checkpoint uses unsupported {name}={config.get(name)!r}"
                )
        model = cls(
            global_hidden_dim=int(config.get("global_hidden_dim", 128)),
            local_hidden_dim=int(config.get("local_hidden_dim", 256)),
        )
        state_dict = {
            key: value
            for key, value in dict(payload["state_dict"]).items()
            if "support_" not in key
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Calibrator checkpoint state mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return model, dict(payload.get("metadata", {}))


__all__ = [
    "DeterministicTransitionCalibrator",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
]
