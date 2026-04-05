#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

ensure_mlx_model

require_file "$ROOT_DIR/datasets/flickr30k/captions.txt"
require_file "$ROOT_DIR/datasets/screen2words_train/manifest.jsonl"
require_file "$ROOT_DIR/datasets/screen2words_val/manifest.jsonl"
require_file "$ROOT_DIR/datasets/private_gallery_local/private_adapt_data.jsonl"

PUBLIC_RUN_NAME="${PUBLIC_RUN_NAME:-semanticgallery_public_stage1}"
FINAL_RUN_NAME="${FINAL_RUN_NAME:-semanticgallery_public_plus_private_data}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-1}"
MAX_EPOCHS_STAGE2="${MAX_EPOCHS_STAGE2:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bfloat16}"
TEXT_UNFREEZE_LAST_N="${TEXT_UNFREEZE_LAST_N:-2}"
VISION_UNFREEZE_LAST_N="${VISION_UNFREEZE_LAST_N:-2}"
LR_STAGE1="${LR_STAGE1:-1e-5}"
LR_STAGE2="${LR_STAGE2:-5e-6}"
PRIVATE_BATCH_SIZE="${PRIVATE_BATCH_SIZE:-8}"
PRIVATE_REPEATS_PER_EPOCH="${PRIVATE_REPEATS_PER_EPOCH:-2}"
PRIVATE_INSTANCE_WEIGHT="${PRIVATE_INSTANCE_WEIGHT:-0.3}"
PRIVATE_DISTILL_WEIGHT="${PRIVATE_DISTILL_WEIGHT:-0.15}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-}"

PUBLIC_RUN_DIR="$ROOT_DIR/logs/$PUBLIC_RUN_NAME"
FINAL_RUN_DIR="$ROOT_DIR/logs/$FINAL_RUN_NAME"
PUBLIC_MANIFEST_PATHS="$ROOT_DIR/datasets/screen2words_train/manifest.jsonl,$ROOT_DIR/datasets/screen2words_val/manifest.jsonl"
FINAL_MANIFEST_PATHS="$PUBLIC_MANIFEST_PATHS,$ROOT_DIR/datasets/private_gallery_local/private_adapt_data.jsonl"

common_args=(
  --model-path "$MLX_MODEL_DIR"
  --dataset-path "$ROOT_DIR/datasets/flickr30k"
  --batch-size "$TRAIN_BATCH_SIZE"
  --precision "$TRAIN_PRECISION"
  --text-unfreeze-last-n "$TEXT_UNFREEZE_LAST_N"
  --vision-unfreeze-last-n "$VISION_UNFREEZE_LAST_N"
)

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  common_args+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi
if [[ -n "$MAX_VAL_STEPS" ]]; then
  common_args+=(--max-val-steps "$MAX_VAL_STEPS")
fi

"$PYTHON_BIN_PATH" "$ROOT_DIR/tools/train_mlx_siglip2.py" \
  "${common_args[@]}" \
  --manifest-paths "$PUBLIC_MANIFEST_PATHS" \
  --epochs "$MAX_EPOCHS_STAGE1" \
  --lr "$LR_STAGE1" \
  --run-dir "$PUBLIC_RUN_DIR"

PUBLIC_WEIGHTS_FILE_PATH="$PUBLIC_RUN_DIR/weights.safetensors"
require_file "$PUBLIC_WEIGHTS_FILE_PATH"

"$PYTHON_BIN_PATH" "$ROOT_DIR/tools/train_mlx_siglip2.py" \
  "${common_args[@]}" \
  --init-weights "$PUBLIC_WEIGHTS_FILE_PATH" \
  --manifest-paths "$FINAL_MANIFEST_PATHS" \
  --epochs "$MAX_EPOCHS_STAGE2" \
  --lr "$LR_STAGE2" \
  --freeze-text-tower \
  --private-batch-size "$PRIVATE_BATCH_SIZE" \
  --private-repeats-per-epoch "$PRIVATE_REPEATS_PER_EPOCH" \
  --public-loss-weight 1.0 \
  --private-instance-weight "$PRIVATE_INSTANCE_WEIGHT" \
  --private-distill-weight "$PRIVATE_DISTILL_WEIGHT" \
  --run-dir "$FINAL_RUN_DIR"

FINAL_WEIGHTS_FILE_PATH="$FINAL_RUN_DIR/weights.safetensors"
require_file "$FINAL_WEIGHTS_FILE_PATH"
printf 'weights=%s\n' "$FINAL_WEIGHTS_FILE_PATH"
