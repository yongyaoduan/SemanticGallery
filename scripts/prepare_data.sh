#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

ensure_env

PRIVATE_GALLERY_PATH="${PRIVATE_GALLERY_PATH:-${1:-}}"
ALIASES_JSON="${ALIASES_JSON:-$ROOT/datasets/private_gallery_local/aliases.local.json}"
QUERY_SUITE_PATH="${QUERY_SUITE_PATH:-$ROOT/datasets/private_gallery_local/query_suite.local.json}"
PRIVATE_DIR="$ROOT/datasets/private_gallery_local"
PREPARE_PUBLIC_DATA="${PREPARE_PUBLIC_DATA:-0}"
MIN_FLICKR_IMAGES="${MIN_FLICKR_IMAGES:-30000}"
MIN_SCREEN2WORDS_TRAIN_ROWS="${MIN_SCREEN2WORDS_TRAIN_ROWS:-15000}"
MIN_SCREEN2WORDS_VAL_ROWS="${MIN_SCREEN2WORDS_VAL_ROWS:-2000}"

count_flickr_images() {
  "$PY" - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)

names = set()
with open(path, "r", encoding="utf-8") as handle:
    next(handle, None)
    for line in handle:
        if line.strip():
            names.add(line.split(",", 1)[0])
print(len(names))
PY
}

count_jsonl_rows() {
  "$PY" - "$1" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)

with open(path, "r", encoding="utf-8") as handle:
    print(sum(1 for line in handle if line.strip()))
PY
}

needs_flickr_refresh() {
  local captions_path="$ROOT/datasets/flickr30k/captions.txt"
  [[ ! -f "$captions_path" ]] && return 0
  local image_count
  image_count="$(count_flickr_images "$captions_path")"
  [[ "$image_count" -lt "$MIN_FLICKR_IMAGES" ]]
}

needs_manifest_refresh() {
  local manifest_path="$1"
  local min_rows="$2"
  [[ ! -f "$manifest_path" ]] && return 0
  local row_count
  row_count="$(count_jsonl_rows "$manifest_path")"
  [[ "$row_count" -lt "$min_rows" ]]
}

if [[ "$PREPARE_PUBLIC_DATA" == "1" ]]; then
  log_step "Preparing public training data"
  if [[ "${FORCE:-0}" == "1" ]] || needs_flickr_refresh; then
      log_step "Refreshing Flickr30k"
      "$PY" "$ROOT/tools/prepare_flickr30k.py" --output-dir "$ROOT/datasets/flickr30k"
  else
      log_step "Reusing Flickr30k"
      log_kv "captions=$ROOT/datasets/flickr30k/captions.txt"
  fi

  if [[ "${FORCE:-0}" == "1" ]] || needs_manifest_refresh "$ROOT/datasets/screen2words_train/manifest.jsonl" "$MIN_SCREEN2WORDS_TRAIN_ROWS"; then
      log_step "Refreshing Screen2Words train"
      "$PY" "$ROOT/tools/prepare_screen2words.py" \
      --split train \
      --output-dir "$ROOT/datasets/screen2words_train"
  else
      log_step "Reusing Screen2Words train"
      log_kv "manifest=$ROOT/datasets/screen2words_train/manifest.jsonl"
  fi

  if [[ "${FORCE:-0}" == "1" ]] || needs_manifest_refresh "$ROOT/datasets/screen2words_val/manifest.jsonl" "$MIN_SCREEN2WORDS_VAL_ROWS"; then
      log_step "Refreshing Screen2Words val"
      "$PY" "$ROOT/tools/prepare_screen2words.py" \
      --split val \
      --output-dir "$ROOT/datasets/screen2words_val"
  else
      log_step "Reusing Screen2Words val"
      log_kv "manifest=$ROOT/datasets/screen2words_val/manifest.jsonl"
  fi
else
  log_step "Skipping full public data download"
fi

if [[ -n "$PRIVATE_GALLERY_PATH" ]]; then
  require_dir "$PRIVATE_GALLERY_PATH"
  log_step "Building local gallery manifest"
  log_kv "private_gallery_path=$(cd "$PRIVATE_GALLERY_PATH" && pwd)"

  gallery_cmd=(
    "$PY" "$ROOT/tools/prepare_gallery_manifest.py"
    --gallery-path "$PRIVATE_GALLERY_PATH"
    --output-path "$PRIVATE_DIR/full_manifest.jsonl"
    --source-name private_gallery_local
  )
  if [[ -f "$ALIASES_JSON" ]]; then
    log_kv "aliases_json=$ALIASES_JSON"
    gallery_cmd+=(--aliases-json "$ALIASES_JSON")
  fi
  "${gallery_cmd[@]}"

  log_step "Sampling capped private adaptation set"
  log_kv "target_size=100"
  adapt_cmd=(
    "$PY" "$ROOT/tools/prepare_private_adapt_manifest.py"
    --source-manifest "$PRIVATE_DIR/full_manifest.jsonl"
    --target-size 100
    --output-path "$PRIVATE_DIR/private_adapt_data.jsonl"
  )
  if [[ -f "$QUERY_SUITE_PATH" ]]; then
    log_kv "query_suite=$QUERY_SUITE_PATH"
    adapt_cmd+=(--query-suite "$QUERY_SUITE_PATH")
  fi
  "${adapt_cmd[@]}"
elif [[ ! -f "$PRIVATE_DIR/full_manifest.jsonl" || ! -f "$PRIVATE_DIR/private_adapt_data.jsonl" ]]; then
  die "set PRIVATE_GALLERY_PATH to build the capped private adaptation set."
fi

printf 'data_ready=%s\n' "$ROOT/datasets"
