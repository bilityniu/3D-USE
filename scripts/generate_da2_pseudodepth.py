#!/usr/bin/env python3
"""Generate the relative DA2 pseudo-depth consumed by 3D-USE Stage 1.

The output is one 16-bit grayscale PNG per source image. It is a relative,
disparity-like prior (larger values indicate nearer content), not metric depth.
Each prediction is robustly normalized by its 1st and 99th percentiles, matching
the pseudo-depth representation used by the released training configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ENCODER_CONFIGS: dict[str, dict[str, Any]] = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}
SCHEMA_NAME = "3duse_da2_pseudodepth"
SCHEMA_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing the RGB images used by COLMAP/3D-USE.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory, normally SCENE/depth.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Official Depth Anything V2 checkpoint.",
    )
    parser.add_argument(
        "--da2-repo",
        type=Path,
        default=None,
        help=(
            "Optional Depth-Anything-V2 repository root. Omit it when "
            "depth_anything_v2 is already importable."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=tuple(ENCODER_CONFIGS),
        default="vitl",
        help="DA2 encoder matching the checkpoint (default: vitl).",
    )
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, or 'auto' (default).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNGs and manifest.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No supported images found in {directory}")
    stems: dict[str, Path] = {}
    for path in images:
        previous = stems.get(path.stem)
        if previous is not None:
            raise ValueError(
                "Output-name collision for image stem "
                f"{path.stem!r}: {previous.name} and {path.name}"
            )
        stems[path.stem] = path
    return images


def load_da2_class(repo: Path | None):
    if repo is not None:
        repo = repo.expanduser().resolve()
        if not (repo / "depth_anything_v2").is_dir():
            raise FileNotFoundError(
                f"{repo} does not contain the depth_anything_v2 package"
            )
        sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("depth_anything_v2.dpt")
    except ImportError as error:
        raise ImportError(
            "Depth Anything V2 is not importable. Pass --da2-repo pointing "
            "to the official Depth-Anything-V2 checkout."
        ) from error
    return module.DepthAnythingV2


def robust_u16(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    valid = finite & (depth > 1e-8)
    if not np.any(valid):
        raise FloatingPointError("DA2 returned no finite positive values")
    lo = float(np.percentile(depth[valid], 1.0))
    hi = float(np.percentile(depth[valid], 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        raise FloatingPointError(
            f"Degenerate DA2 prediction range: q01={lo}, q99={hi}"
        )
    normalized = np.clip((np.nan_to_num(depth, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 65535.0 + 0.5).astype(np.uint16), lo, hi


def main() -> None:
    args = parse_args()
    if args.input_size < 32:
        raise ValueError("--input-size must be at least 32")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing DA2 checkpoint: {checkpoint}")
    images = collect_images(args.images)
    output = args.output.expanduser()
    manifest_path = output / "manifest.json"
    collisions = [
        output / f"{path.stem}.png"
        for path in images
        if (output / f"{path.stem}.png").exists()
    ]
    if not args.overwrite and (manifest_path.exists() or collisions):
        first = manifest_path if manifest_path.exists() else collisions[0]
        raise FileExistsError(
            f"Output already exists: {first}. Pass --overwrite to replace it."
        )
    output.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    DepthAnythingV2 = load_da2_class(args.da2_repo)
    model = DepthAnythingV2(**ENCODER_CONFIGS[args.encoder])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state-dict checkpoint, got {type(state).__name__}")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Failed to decode image: {path}")
            prediction = np.asarray(
                model.infer_image(image, int(args.input_size)),
                dtype=np.float32,
            )
            if prediction.shape != image.shape[:2]:
                prediction = cv2.resize(
                    prediction,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            depth_u16, q01, q99 = robust_u16(prediction)
            destination = output / f"{path.stem}.png"
            if not cv2.imwrite(str(destination), depth_u16):
                raise OSError(f"Failed to write pseudo-depth: {destination}")
            valid = prediction[np.isfinite(prediction)]
            records.append(
                {
                    "image": path.name,
                    "pseudo_depth": destination.name,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "prediction_min": float(valid.min()),
                    "prediction_max": float(valid.max()),
                    "normalization_q01": q01,
                    "normalization_q99": q99,
                }
            )
            print(
                f"[{index:04d}/{len(images):04d}] "
                f"{path.name} -> {destination.name}",
                flush=True,
            )

    manifest = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "representation": "relative_disparity_like_pseudodepth",
        "metric_depth": False,
        "larger_values_are_nearer": True,
        "storage": "uint16_grayscale_png",
        "normalization": "per_image_percentile_1_99",
        "images": args.images.name,
        "output": output.name,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        "encoder": args.encoder,
        "input_size": int(args.input_size),
        "device": str(device),
        "count": len(records),
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
