#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Run the final 3D-USE Stage-1 and Stage-2 pipeline for one scene.

Usage:
  run_two_stage.sh --data SCENE_DIR --output OUTPUT_DIR \
    --calibrator FILE --uie-proposer FILE [options]

Required:
  --data PATH          COLMAP scene root.
  --output PATH        Output root for both stages.
  --calibrator PATH    Trained transition calibrator checkpoint.
  --uie-proposer PATH  Frozen UIE proposer checkpoint.

Options:
  --images-path PATH   Image directory relative to the scene root (default: images_wb).
  --depths-path PATH   DA2 pseudo-depth directory (default: depth).
  --colmap-path PATH   COLMAP model directory (default: colmap/sparse/0).
  --stage1-steps N     Stage-1 steps (default: 15000).
  --stage2-steps N     Additional Stage-2 steps (default: 5000).
  --project NAME       TensorBoard/W&B project name (default: 3D-USE).
  --vis BACKEND        tensorboard, wandb, viewer, viewer+tensorboard, or
                       viewer+wandb (default: tensorboard).
  --gpu ID             Set CUDA_VISIBLE_DEVICES for both stages.
  --ns-train PATH      ns-train executable (default: ns-train from PATH).
  --reuse-existing     Explicitly reuse completed checkpoints in OUTPUT_DIR.
  -h, --help           Show this help.

Existing checkpoints are never reused without --reuse-existing. An incomplete
stage is not overwritten; choose a fresh output directory in that case.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "Missing value for $1" >&2
    usage >&2
    exit 2
  fi
}

DATA=""
OUTPUT=""
CALIBRATOR=""
UIE_PROPOSER=""
IMAGES_PATH="images_wb"
DEPTHS_PATH="depth"
COLMAP_PATH="colmap/sparse/0"
STAGE1_STEPS=15000
STAGE2_STEPS=5000
PROJECT="3D-USE"
VIS="tensorboard"
GPU=""
NS_TRAIN="ns-train"
REUSE_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) require_value "$@"; DATA="$2"; shift 2 ;;
    --output) require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --calibrator) require_value "$@"; CALIBRATOR="$2"; shift 2 ;;
    --uie-proposer) require_value "$@"; UIE_PROPOSER="$2"; shift 2 ;;
    --images-path) require_value "$@"; IMAGES_PATH="$2"; shift 2 ;;
    --depths-path) require_value "$@"; DEPTHS_PATH="$2"; shift 2 ;;
    --colmap-path) require_value "$@"; COLMAP_PATH="$2"; shift 2 ;;
    --stage1-steps) require_value "$@"; STAGE1_STEPS="$2"; shift 2 ;;
    --stage2-steps) require_value "$@"; STAGE2_STEPS="$2"; shift 2 ;;
    --project) require_value "$@"; PROJECT="$2"; shift 2 ;;
    --vis) require_value "$@"; VIS="$2"; shift 2 ;;
    --gpu) require_value "$@"; GPU="$2"; shift 2 ;;
    --ns-train) require_value "$@"; NS_TRAIN="$2"; shift 2 ;;
    --reuse-existing) REUSE_EXISTING=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA" ]] || { echo "--data is required" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
[[ -n "$CALIBRATOR" ]] || { echo "--calibrator is required" >&2; exit 2; }
[[ -n "$UIE_PROPOSER" ]] || {
  echo "--uie-proposer is required" >&2
  exit 2
}
[[ "$STAGE1_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--stage1-steps must be a positive integer" >&2
  exit 2
}
[[ "$STAGE2_STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--stage2-steps must be a positive integer" >&2
  exit 2
}

COMMON=(
  --data "$DATA"
  --output "$OUTPUT"
  --images-path "$IMAGES_PATH"
  --depths-path "$DEPTHS_PATH"
  --colmap-path "$COLMAP_PATH"
  --project "$PROJECT"
  --vis "$VIS"
  --ns-train "$NS_TRAIN"
)
if [[ -n "$GPU" ]]; then
  COMMON+=(--gpu "$GPU")
fi

STAGE1_CHECKPOINT_DIR="$OUTPUT/stage1/3duse-stage1/run/nerfstudio_models"
STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT_DIR/step-$(printf '%09d' "$STAGE1_STEPS").ckpt"
if [[ ! -f "$STAGE1_CHECKPOINT" ]]; then
  if [[ -e "$OUTPUT/stage1/3duse-stage1/run/config.yml" ]]; then
    echo "Incomplete Stage-1 output exists: $OUTPUT/stage1/3duse-stage1/run" >&2
    exit 1
  fi
  "$SCRIPT_DIR/train_stage1.sh" "${COMMON[@]}" \
    --steps "$STAGE1_STEPS" --run-name stage1
else
  if [[ "$REUSE_EXISTING" -ne 1 ]]; then
    echo "Stage-1 checkpoint already exists: $STAGE1_CHECKPOINT" >&2
    echo "Use --reuse-existing only if it belongs to this scene and configuration." >&2
    exit 1
  fi
  echo "Reusing Stage-1 checkpoint: $STAGE1_CHECKPOINT"
fi

STAGE2_END_STEP="$((STAGE1_STEPS + STAGE2_STEPS))"
STAGE2_CHECKPOINT_DIR="$OUTPUT/stage2/3duse-stage2/run/nerfstudio_models"
STAGE2_CHECKPOINT="$STAGE2_CHECKPOINT_DIR/step-$(printf '%09d' "$STAGE2_END_STEP").ckpt"
if [[ ! -f "$STAGE2_CHECKPOINT" ]]; then
  if [[ -e "$OUTPUT/stage2/3duse-stage2/run/config.yml" ]]; then
    echo "Incomplete Stage-2 output exists: $OUTPUT/stage2/3duse-stage2/run" >&2
    exit 1
  fi
  "$SCRIPT_DIR/train_stage2.sh" "${COMMON[@]}" \
    --stage1-checkpoint-dir "$STAGE1_CHECKPOINT_DIR" \
    --stage1-step "$STAGE1_STEPS" \
    --steps "$STAGE2_STEPS" \
    --run-name stage2 \
    --calibrator "$CALIBRATOR" \
    --uie-proposer "$UIE_PROPOSER"
else
  if [[ "$REUSE_EXISTING" -ne 1 ]]; then
    echo "Stage-2 checkpoint already exists: $STAGE2_CHECKPOINT" >&2
    echo "Use --reuse-existing only if it belongs to this scene and configuration." >&2
    exit 1
  fi
  echo "Reusing Stage-2 checkpoint: $STAGE2_CHECKPOINT"
fi

printf 'Stage-1 checkpoint: %s\nStage-2 checkpoint: %s\n' \
  "$STAGE1_CHECKPOINT" "$STAGE2_CHECKPOINT"
