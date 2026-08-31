#!/usr/bin/env python3
"""Compile paired raw/enhanced images into effective transition observations.

The representation is deliberately renderer-agnostic: every pair is reduced
to the same global-affine plus local-diagonal transition that Stage-2 extracts
from its own full underwater renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threeduse.enhance.effective_transition import (  # noqa: E402
    extract_effective_transition,
    global_transition_condition,
    local_transition_condition,
)
from threeduse.enhance.uie_proposer import (  # noqa: E402
    load_frozen_uie_proposer,
    run_frozen_uie_proposer,
)


SCHEMA_NAME = "3duse_effective_transition_dataset"
SCHEMA_VERSION = 4
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="Raw UIEB image directory."
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Filename-aligned paired enhanced target directory.",
    )
    parser.add_argument("--dataset-name", default="paired_uie")
    parser.add_argument(
        "--target-provenance",
        default="paired_enhancement",
        help="Free-form provenance label stored in dataset metadata.",
    )
    parser.add_argument(
        "--uie-proposer-checkpoint",
        type=Path,
        required=True,
        help="Frozen UIE proposer used to construct proposal operators.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output transition dataset."
    )
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--local-grid", type=int, default=16)
    parser.add_argument("--global-ridge", type=float, default=1e-3)
    parser.add_argument("--local-ridge", type=float, default=1e-2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing dataset."
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    images = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUFFIXES
    }
    if not images:
        raise RuntimeError(f"No images found in {directory}")
    return images


def load_image(path: Path) -> torch.Tensor:
    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if array is None:
        raise OSError(f"Failed to decode image: {path}")
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(array).contiguous()


def match_target_size(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if target.shape[:2] == source.shape[:2]:
        return target
    return (
        F.interpolate(
            target.permute(2, 0, 1)[None],
            size=source.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .contiguous()
    )


def build(args: argparse.Namespace) -> dict[str, object]:
    raw = image_map(args.raw_dir)
    target = image_map(args.target_dir)
    names = sorted(set(raw) & set(target))
    if not names:
        raise RuntimeError("Raw and target directories have no matching filenames")
    if set(raw) != set(target):
        print(
            f"warning: using {len(names)} intersections; "
            f"raw-only={len(set(raw) - set(target))}, target-only={len(set(target) - set(raw))}",
            flush=True,
        )
    if not args.uie_proposer_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing UIE proposer: {args.uie_proposer_checkpoint}"
        )
    device = resolve_device(args.device)
    proposer_model = load_frozen_uie_proposer(
        args.uie_proposer_checkpoint,
        device=device,
    )

    global_operator: list[torch.Tensor] = []
    global_condition: list[torch.Tensor] = []
    local_operator: list[torch.Tensor] = []
    local_condition: list[torch.Tensor] = []
    local_confidence: list[torch.Tensor] = []
    local_image_index: list[torch.Tensor] = []
    fit_summary: list[dict[str, float]] = []

    with torch.no_grad():
        for pair_index, name in enumerate(names):
            raw_image = load_image(raw[name])
            target_image = match_target_size(load_image(target[name]), raw_image)
            proposed_image = run_frozen_uie_proposer(
                proposer_model,
                raw_image.to(device),
                max_side=int(args.max_side),
            ).cpu()
            transition = extract_effective_transition(
                raw_image,
                target_image,
                local_grid_long_side=int(args.local_grid),
                max_side=int(args.max_side),
                global_ridge=float(args.global_ridge),
                local_ridge=float(args.local_ridge),
            )
            uie_proposal = extract_effective_transition(
                raw_image,
                proposed_image,
                local_grid_long_side=int(args.local_grid),
                max_side=int(args.max_side),
                global_ridge=float(args.global_ridge),
                local_ridge=float(args.local_ridge),
            )
            global_cond = global_transition_condition(transition, uie_proposal)
            local_cond_grid = local_transition_condition(
                transition,
                uie_proposal,
                transition.global_operator,
            )
            local_op = transition.local_operator.reshape(
                -1, transition.local_operator.shape[-1]
            )
            local_cond = local_cond_grid.reshape(
                -1, local_cond_grid.shape[-1]
            )
            confidence = transition.local_confidence.reshape(-1)
            valid = (
                torch.isfinite(local_op).all(dim=-1)
                & torch.isfinite(local_cond).all(dim=-1)
                & torch.isfinite(confidence)
            )
            global_operator.append(transition.global_operator[0].cpu())
            global_condition.append(global_cond[0].cpu())
            local_operator.append(local_op[valid].cpu())
            local_condition.append(local_cond[valid].cpu())
            local_confidence.append(confidence[valid].cpu())
            local_image_index.append(
                torch.full((int(valid.sum()),), pair_index, dtype=torch.long)
            )
            fit_summary.append(
                {
                    "pair_index": pair_index,
                    "global_mse": float(transition.global_residual[0]),
                    "local_mse": float(transition.local_residual.mean()),
                    "proposal_global_mse": float(uie_proposal.global_residual[0]),
                    "proposal_local_mse": float(uie_proposal.local_residual.mean()),
                    "proposal_global_magnitude": float(
                        uie_proposal.global_operator.norm(dim=-1).mean()
                    ),
                    "target_global_magnitude": float(
                        transition.global_operator.norm(dim=-1).mean()
                    ),
                    "local_confidence": (
                        float(confidence[valid].mean()) if valid.any() else 0.0
                    ),
                }
            )
            if pair_index == 0 or (pair_index + 1) % 50 == 0 or pair_index + 1 == len(names):
                print(f"[{pair_index + 1:04d}/{len(names):04d}] {name}", flush=True)

    global_operator_tensor = torch.stack(global_operator).float()
    global_condition_tensor = torch.stack(global_condition).float()
    local_operator_tensor = torch.cat(local_operator).float()
    local_condition_tensor = torch.cat(local_condition).float()
    local_confidence_tensor = torch.cat(local_confidence).float().clamp(0.0, 1.0)
    if not all(
        torch.isfinite(value).all()
        for value in (
            global_operator_tensor,
            global_condition_tensor,
            local_operator_tensor,
            local_condition_tensor,
            local_confidence_tensor,
        )
    ):
        raise FloatingPointError("Non-finite compiled transitions")

    metadata = {
        "raw_dir": args.raw_dir.name,
        "target_dir": args.target_dir.name,
        "dataset_name": str(args.dataset_name),
        "target_provenance": str(args.target_provenance),
        "uie_proposer_checkpoint": args.uie_proposer_checkpoint.name,
        "uie_proposer_checkpoint_sha256": file_sha256(
            args.uie_proposer_checkpoint
        ),
        "num_pairs": len(names),
        "num_transitions": len(global_operator),
        "num_local_observations": int(local_operator_tensor.shape[0]),
        "max_side": int(args.max_side),
        "local_grid": int(args.local_grid),
        "global_ridge": float(args.global_ridge),
        "local_ridge": float(args.local_ridge),
        "device": str(device),
        "pair_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "representation": "paired_raw_to_target_operator_conditioned_on_uie_proposal",
        "mean_global_fit_mse": float(np.mean([entry["global_mse"] for entry in fit_summary])),
        "mean_local_fit_mse": float(np.mean([entry["local_mse"] for entry in fit_summary])),
    }
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "global_operator": global_operator_tensor,
        "global_condition": global_condition_tensor,
        "local_operator": local_operator_tensor,
        "local_condition": local_condition_tensor,
        "local_confidence": local_confidence_tensor,
        "local_image_index": torch.cat(local_image_index),
        "pair_names": names,
        "fit_summary": fit_summary,
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    if args.max_side < 8 or args.local_grid < 2:
        raise ValueError("max-side must be >=8 and local-grid >=2")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Pass --overwrite to replace it."
        )
    payload = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(payload["metadata"], indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **payload["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
