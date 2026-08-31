#!/usr/bin/env python3
"""Train the deterministic operator calibrator on paired UIE transitions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threeduse.enhance.effective_transition import (  # noqa: E402
    GLOBAL_CONDITION_DIM,
    GLOBAL_OPERATOR_DIM,
    LOCAL_CONDITION_DIM,
    LOCAL_OPERATOR_DIM,
)
from threeduse.enhance.transition_calibrator import (  # noqa: E402
    DeterministicTransitionCalibrator,
)


DATASET_SCHEMA = "3duse_effective_transition_dataset"
DATASET_VERSION = 4
CONDITION_MODE = "source_and_proposal"
PREDICTION_MODE = "proposal_residual"
DEFAULT_TRAINING_STEPS = 750
LOG_EVERY = 250


@dataclass
class PairedTransitions:
    path: Path
    payload: dict

    @property
    def name(self) -> str:
        metadata = self.payload.get("metadata", {})
        return str(metadata.get("dataset_name", self.path.stem))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_TRAINING_STEPS,
        help=(
            "Optimization steps. The default reproduces the released training "
            f"budget ({DEFAULT_TRAINING_STEPS})."
        ),
    )
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--local-batch", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--global-hidden", type=int, default=128)
    parser.add_argument("--local-hidden", type=int, default=256)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.5)
    parser.add_argument("--normalization-local-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing checkpoint."
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def portable_dataset_metadata(dataset: PairedTransitions) -> dict:
    """Remove host-specific paths before embedding provenance in a checkpoint."""

    metadata = dict(dataset.payload.get("metadata", {}))
    for key in ("raw_dir", "target_dir", "uie_proposer_checkpoint"):
        value = metadata.get(key)
        if value:
            metadata[key] = Path(str(value)).name
    return metadata


def _load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping dataset at {path}")
    if (
        payload.get("schema") != DATASET_SCHEMA
        or int(payload.get("schema_version", -1)) != DATASET_VERSION
    ):
        raise ValueError(
            f"Unsupported dataset {payload.get('schema')!r} "
            f"version {payload.get('schema_version')!r} at {path}"
        )
    required = (
        "global_operator",
        "global_condition",
        "local_operator",
        "local_condition",
        "local_confidence",
        "local_image_index",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Dataset {path} is missing {missing}")
    return dict(payload)


def load_dataset(path: Path) -> PairedTransitions:
    payload = _load_payload(path)
    num_pairs = int(payload["global_operator"].shape[0])
    image_index = payload["local_image_index"].long()
    if num_pairs < 1 or image_index.numel() < 1:
        raise ValueError(f"Transition dataset is empty: {path}")
    if int(image_index.min()) < 0 or int(image_index.max()) >= num_pairs:
        raise ValueError(f"Local observations reference invalid image rows: {path}")
    return PairedTransitions(path, payload)


def sample_rows(
    rows: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    if rows.numel() <= count:
        return rows
    return rows[
        torch.randint(rows.numel(), (count,), generator=generator)
    ]


def global_rows(dataset: PairedTransitions) -> torch.Tensor:
    return torch.arange(dataset.payload["global_operator"].shape[0])


def local_rows(dataset: PairedTransitions) -> torch.Tensor:
    return torch.arange(dataset.payload["local_operator"].shape[0])


def _local_condition_with_predicted_global(
    model: DeterministicTransitionCalibrator,
    dataset: PairedTransitions,
    local_rows: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    data = dataset.payload
    image_rows = data["local_image_index"][local_rows].long()
    global_condition = data["global_condition"][image_rows].to(device)
    predicted_global = model.global_target(global_condition).detach()
    local_condition = data["local_condition"][local_rows].to(device).clone()
    local_condition[..., -GLOBAL_OPERATOR_DIM:] = predicted_global
    return local_condition


def dataset_loss(
    model: DeterministicTransitionCalibrator,
    dataset: PairedTransitions,
    global_rows: torch.Tensor,
    local_rows: torch.Tensor,
    *,
    beta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = dataset.payload
    global_condition = data["global_condition"][global_rows].to(device)
    global_target = data["global_operator"][global_rows].to(device)
    global_prediction = model.global_target(global_condition)
    global_loss = F.smooth_l1_loss(
        model.normalize_global_operator(global_prediction),
        model.normalize_global_operator(global_target),
        beta=beta,
    )

    local_condition = _local_condition_with_predicted_global(
        model, dataset, local_rows, device
    )
    local_target = data["local_operator"][local_rows].to(device)
    local_prediction = model.local_target(local_condition)
    per_local = F.smooth_l1_loss(
        model.normalize_local_operator(local_prediction),
        model.normalize_local_operator(local_target),
        beta=beta,
        reduction="none",
    ).mean(dim=-1)
    confidence = data["local_confidence"][local_rows].to(device)
    local_loss = (confidence * per_local).sum() / confidence.sum().clamp_min(1e-6)
    return global_loss, local_loss


def _normalization_samples(
    dataset: PairedTransitions,
    *,
    local_limit: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled_global_rows = global_rows(dataset)
    sampled_local_rows = sample_rows(
        local_rows(dataset), local_limit, generator
    )
    data = dataset.payload
    global_condition = data["global_condition"][sampled_global_rows]
    global_operator = data["global_operator"][sampled_global_rows]
    global_operator = global_operator - global_condition[
        ..., -GLOBAL_OPERATOR_DIM:
    ]
    local_condition = data["local_condition"][sampled_local_rows]
    local_operator = data["local_operator"][sampled_local_rows]
    proposal_start = (
        LOCAL_CONDITION_DIM + GLOBAL_CONDITION_DIM + GLOBAL_OPERATOR_DIM
    )
    local_operator = local_operator - local_condition[
        ..., proposal_start : proposal_start + LOCAL_OPERATOR_DIM
    ]
    return {
        "global_operator": global_operator.float(),
        "global_condition": global_condition.float(),
        "local_operator": local_operator.float(),
        "local_condition": local_condition.float(),
    }


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.global_batch < 1 or args.local_batch < 1:
        raise ValueError("Steps and batch sizes must be positive")
    if args.smooth_l1_beta <= 0:
        raise ValueError("--smooth-l1-beta must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Pass --overwrite to replace it."
        )
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Missing transition dataset: {args.dataset}")
    device = resolve_device(args.device)
    dataset = load_dataset(args.dataset)

    model = DeterministicTransitionCalibrator(
        global_hidden_dim=args.global_hidden,
        local_hidden_dim=args.local_hidden,
        condition_mode=CONDITION_MODE,
        prediction_mode=PREDICTION_MODE,
    ).to(device)
    normalization = _normalization_samples(
        dataset,
        local_limit=args.normalization_local_samples,
        seed=args.seed + 31,
    )
    model.set_normalization(**normalization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.learning_rate * 0.05,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    history: list[dict[str, object]] = []

    for step in range(1, args.steps + 1):
        model.train()
        sampled_global_rows = sample_rows(
            global_rows(dataset), args.global_batch, generator
        )
        sampled_local_rows = sample_rows(
            local_rows(dataset), args.local_batch, generator
        )
        global_loss, local_loss = dataset_loss(
            model,
            dataset,
            sampled_global_rows,
            sampled_local_rows,
            beta=args.smooth_l1_beta,
            device=device,
        )
        loss = global_loss + local_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite calibrator loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % LOG_EVERY == 0 or step == args.steps:
            entry = {
                "step": step,
                "train_global": float(global_loss.detach()),
                "train_local": float(local_loss.detach()),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)
    model.to("cpu")
    metadata = {
        "dataset": dataset.path.name,
        "dataset_metadata": portable_dataset_metadata(dataset),
        "dataset_name": dataset.name,
        "num_train_pairs": int(global_rows(dataset).numel()),
        "sampling": "uniform_full_dataset",
        "seed": args.seed,
        "steps": args.steps,
        "smooth_l1_beta": args.smooth_l1_beta,
        "history": history,
        "objective": "normalized_smooth_l1_operator_calibration",
        "condition_mode": CONDITION_MODE,
        "prediction_mode": PREDICTION_MODE,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint_payload(metadata), args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
