#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Train the final 3D-USE Stage-2 scene-level appearance model.

Usage:
  train_stage2.sh --data SCENE_DIR --output OUTPUT_DIR \
    --stage1-checkpoint-dir DIR --calibrator FILE --uie-proposer FILE [options]

Required:
  --data PATH                   COLMAP scene root.
  --output PATH                 Nerfstudio output root.
  --stage1-checkpoint-dir PATH  Directory containing the Stage-1 checkpoint.
  --calibrator PATH             Trained transition calibrator checkpoint.
  --uie-proposer PATH           Frozen UIE proposer checkpoint.

Options:
  --images-path PATH   Image directory relative to the scene root (default: images_wb).
  --depths-path PATH   DA2 pseudo-depth directory (default: depth).
  --colmap-path PATH   COLMAP model directory (default: colmap/sparse/0).
  --stage1-step N      Stage-1 checkpoint step (default: 15000).
  --steps N            Additional Stage-2 steps (default: 5000).
  --run-name NAME      Experiment name below OUTPUT_DIR (default: stage2).
  --project NAME       TensorBoard/W&B project name (default: 3D-USE).
  --vis BACKEND        tensorboard, wandb, viewer, viewer+tensorboard, or
                       viewer+wandb (default: tensorboard).
  --gpu ID             Set CUDA_VISIBLE_DEVICES for this process.
  --ns-train PATH      ns-train executable (default: ns-train from PATH).
  -h, --help           Show this help.
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
STAGE1_CHECKPOINT_DIR=""
CALIBRATOR=""
UIE_PROPOSER=""
IMAGES_PATH="images_wb"
DEPTHS_PATH="depth"
COLMAP_PATH="colmap/sparse/0"
STAGE1_STEP=15000
STEPS=5000
RUN_NAME="stage2"
PROJECT="3D-USE"
VIS="tensorboard"
GPU=""
NS_TRAIN="ns-train"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) require_value "$@"; DATA="$2"; shift 2 ;;
    --output) require_value "$@"; OUTPUT="$2"; shift 2 ;;
    --stage1-checkpoint-dir) require_value "$@"; STAGE1_CHECKPOINT_DIR="$2"; shift 2 ;;
    --calibrator) require_value "$@"; CALIBRATOR="$2"; shift 2 ;;
    --uie-proposer) require_value "$@"; UIE_PROPOSER="$2"; shift 2 ;;
    --images-path) require_value "$@"; IMAGES_PATH="$2"; shift 2 ;;
    --depths-path) require_value "$@"; DEPTHS_PATH="$2"; shift 2 ;;
    --colmap-path) require_value "$@"; COLMAP_PATH="$2"; shift 2 ;;
    --stage1-step) require_value "$@"; STAGE1_STEP="$2"; shift 2 ;;
    --steps) require_value "$@"; STEPS="$2"; shift 2 ;;
    --run-name) require_value "$@"; RUN_NAME="$2"; shift 2 ;;
    --project) require_value "$@"; PROJECT="$2"; shift 2 ;;
    --vis) require_value "$@"; VIS="$2"; shift 2 ;;
    --gpu) require_value "$@"; GPU="$2"; shift 2 ;;
    --ns-train) require_value "$@"; NS_TRAIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA" ]] || { echo "--data is required" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
[[ -n "$STAGE1_CHECKPOINT_DIR" ]] || {
  echo "--stage1-checkpoint-dir is required" >&2
  exit 2
}
[[ -n "$CALIBRATOR" ]] || { echo "--calibrator is required" >&2; exit 2; }
[[ -n "$UIE_PROPOSER" ]] || {
  echo "--uie-proposer is required" >&2
  exit 2
}
[[ "$STAGE1_STEP" =~ ^[1-9][0-9]*$ ]] || {
  echo "--stage1-step must be a positive integer" >&2
  exit 2
}
[[ "$STEPS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--steps must be a positive integer" >&2
  exit 2
}
case "$VIS" in
  tensorboard|wandb|viewer|viewer+tensorboard|viewer+wandb) ;;
  *) echo "Unsupported --vis backend: $VIS" >&2; exit 2 ;;
esac

[[ -d "$DATA" ]] || { echo "Missing scene root: $DATA" >&2; exit 1; }
[[ -d "$DATA/$IMAGES_PATH" ]] || {
  echo "Missing image directory: $DATA/$IMAGES_PATH" >&2
  exit 1
}
[[ -d "$DATA/$DEPTHS_PATH" ]] || {
  echo "Missing DA2 pseudo-depth directory: $DATA/$DEPTHS_PATH" >&2
  exit 1
}
[[ -d "$DATA/$COLMAP_PATH" ]] || {
  echo "Missing COLMAP model: $DATA/$COLMAP_PATH" >&2
  exit 1
}
[[ -f "$CALIBRATOR" ]] || {
  echo "Missing transition calibrator: $CALIBRATOR" >&2
  exit 1
}
[[ -f "$UIE_PROPOSER" ]] || {
  echo "Missing UIE proposer: $UIE_PROPOSER" >&2
  exit 1
}
STAGE1_CHECKPOINT="$STAGE1_CHECKPOINT_DIR/step-$(printf '%09d' "$STAGE1_STEP").ckpt"
[[ -f "$STAGE1_CHECKPOINT" ]] || {
  echo "Missing Stage-1 checkpoint: $STAGE1_CHECKPOINT" >&2
  exit 1
}
command -v "$NS_TRAIN" >/dev/null 2>&1 || {
  echo "ns-train executable not found: $NS_TRAIN" >&2
  exit 1
}

END_STEP="$((STAGE1_STEP + STEPS))"
INTERNAL_END_STEP="$((END_STEP + 1))"
BASE_DIR="$OUTPUT/$RUN_NAME/3duse-stage2/run"
CHECKPOINT_DIR="$BASE_DIR/nerfstudio_models"
FINAL_CHECKPOINT="$CHECKPOINT_DIR/step-$(printf '%09d' "$END_STEP").ckpt"
if [[ -e "$BASE_DIR/config.yml" || -e "$FINAL_CHECKPOINT" ]]; then
  echo "Stage-2 output already exists: $BASE_DIR" >&2
  echo "Choose a new --output/--run-name, or use run_two_stage.sh to reuse a completed stage." >&2
  exit 1
fi

mkdir -p "$OUTPUT"
export THREEDUSE_STAGE1_NUM_STEPS="$((STAGE1_STEP + 1))"
export THREEDUSE_STAGE2_END_STEP="$INTERNAL_END_STEP"
export THREEDUSE_STAGE2_CALIBRATOR="$CALIBRATOR"
export THREEDUSE_STAGE2_UIE_PROPOSER_CHECKPOINT="$UIE_PROPOSER"
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_JIT=0

COMMAND=(
  "$NS_TRAIN" 3duse-stage2
  --project-name "$PROJECT"
  --experiment-name "$RUN_NAME"
  --timestamp run
  --output-dir "$OUTPUT"
  --vis "$VIS"
  --load-dir "$STAGE1_CHECKPOINT_DIR"
  --load-step "$STAGE1_STEP"
  --load-optimizer False
  --load-scheduler False
  --max-num-iterations "$INTERNAL_END_STEP"
  --steps-per-save "$END_STEP"
  --steps-per-eval-all-images 0
  --pipeline.model.num-steps "$INTERNAL_END_STEP"
  --pipeline.model.stage2-transition-calibrator-checkpoint "$CALIBRATOR"
  --pipeline.model.stage2-uie-proposer-checkpoint "$UIE_PROPOSER"
  colmap
  --data "$DATA"
  --images-path "$IMAGES_PATH"
  --depths-path "$DEPTHS_PATH"
  --colmap-path "$COLMAP_PATH"
  --load-3D-points False
  --downscale-factor 1
)

echo "Stage 2: $STAGE1_CHECKPOINT -> $BASE_DIR"
if [[ -n "$GPU" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "${COMMAND[@]}" 2>&1 | tee "$OUTPUT/${RUN_NAME}.log"
else
  "${COMMAND[@]}" 2>&1 | tee "$OUTPUT/${RUN_NAME}.log"
fi

[[ -f "$FINAL_CHECKPOINT" ]] || {
  echo "Training ended without the expected checkpoint: $FINAL_CHECKPOINT" >&2
  exit 1
}

# Nerfstudio reuses --load-step when evaluating a saved config. During Stage 2
# it points to the Stage-1 initialization checkpoint, so update the completed
# config to load the final Stage-2 checkpoint for ns-render and ns-eval.
CONFIG_PATH="$BASE_DIR/config.yml"
[[ -f "$CONFIG_PATH" ]] || {
  echo "Training ended without the expected config: $CONFIG_PATH" >&2
  exit 1
}
sed -i "s/^load_step: .*/load_step: $END_STEP/" "$CONFIG_PATH"
grep -qx "load_step: $END_STEP" "$CONFIG_PATH" || {
  echo "Failed to finalize checkpoint step in: $CONFIG_PATH" >&2
  exit 1
}

printf 'Stage-2 checkpoint: %s\n' "$FINAL_CHECKPOINT"
