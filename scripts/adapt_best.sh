#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

ensure_mlx_model
ensure_published_stage2_public_anchor

require_file "$ROOT/datasets/private_gallery_local/private_adapt_data.jsonl"
require_file "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR/captions.txt"
require_file "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST"

FINAL_RUN_NAME="${FINAL_RUN_NAME:-semanticgallery_private_data_adapted}"
STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-}"
MAX_EPOCHS_STAGE2="${MAX_EPOCHS_STAGE2:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_PRECISION="${TRAIN_PRECISION:-bfloat16}"
TEXT_UNFREEZE_LAST_N="${TEXT_UNFREEZE_LAST_N:-2}"
VISION_UNFREEZE_LAST_N="${VISION_UNFREEZE_LAST_N:-2}"
LR_STAGE2="${LR_STAGE2:-5e-6}"
PRIVATE_BATCH_SIZE="${PRIVATE_BATCH_SIZE:-8}"
PRIVATE_REPEATS_PER_EPOCH="${PRIVATE_REPEATS_PER_EPOCH:-2}"
PRIVATE_INSTANCE_WEIGHT="${PRIVATE_INSTANCE_WEIGHT:-0.3}"
PRIVATE_DISTILL_WEIGHT="${PRIVATE_DISTILL_WEIGHT:-0.15}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
MAX_VAL_STEPS="${MAX_VAL_STEPS:-}"

FINAL_RUN_DIR="$ROOT/logs/$FINAL_RUN_NAME"
FINAL_MANIFESTS="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST,$ROOT/datasets/private_gallery_local/private_adapt_data.jsonl"
BASE_STAGE1_WEIGHTS="$(resolve_stage1_weights_file "$STAGE1_WEIGHTS")"

common_args=(
  --model-path "$MLX_MODEL_DIR"
  --dataset-path "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR"
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

log_step "Running Stage 2 adaptation"
log_kv "stage1_weights=$BASE_STAGE1_WEIGHTS"
log_kv "run_dir=$FINAL_RUN_DIR"
log_kv "public_anchor_flickr=$PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR"
log_kv "public_anchor_manifest=$PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST"

"$PY" "$ROOT/tools/train_mlx_siglip2.py" \
  "${common_args[@]}" \
  --init-weights "$BASE_STAGE1_WEIGHTS" \
  --manifest-paths "$FINAL_MANIFESTS" \
  --epochs "$MAX_EPOCHS_STAGE2" \
  --lr "$LR_STAGE2" \
  --freeze-text-tower \
  --private-batch-size "$PRIVATE_BATCH_SIZE" \
  --private-repeats-per-epoch "$PRIVATE_REPEATS_PER_EPOCH" \
  --public-loss-weight 1.0 \
  --private-instance-weight "$PRIVATE_INSTANCE_WEIGHT" \
  --private-distill-weight "$PRIVATE_DISTILL_WEIGHT" \
  --run-dir "$FINAL_RUN_DIR"

FINAL_WEIGHTS="$FINAL_RUN_DIR/weights.safetensors"
require_file "$FINAL_WEIGHTS"
printf 'weights=%s\n' "$FINAL_WEIGHTS"
