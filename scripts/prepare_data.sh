#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

ensure_env

PRIVATE_GALLERY_DIR="${PRIVATE_GALLERY_DIR:-${1:-}}"
PRIVATE_DATA_DIR="$ROOT_DIR/datasets/private_gallery_local"
PREPARE_PUBLIC_DATA="${PREPARE_PUBLIC_DATA:-0}"
MIN_FLICKR_IMAGES="${MIN_FLICKR_IMAGES:-30000}"
MIN_SCREEN2WORDS_TRAIN_ROWS="${MIN_SCREEN2WORDS_TRAIN_ROWS:-15000}"
MIN_SCREEN2WORDS_VAL_ROWS="${MIN_SCREEN2WORDS_VAL_ROWS:-2000}"
PRIVATE_ADAPT_TARGET_SIZE=100
PRIVATE_ADAPT_MISSING_RETRAIN_THRESHOLD=0.10

count_flickr_images() {
  "$PYTHON_BIN_PATH" - "$1" <<'PY'
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
  "$PYTHON_BIN_PATH" - "$1" <<'PY'
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
  local captions_file_path="$ROOT_DIR/datasets/flickr30k/captions.txt"
  [[ ! -f "$captions_file_path" ]] && return 0
  local image_count
  image_count="$(count_flickr_images "$captions_file_path")"
  [[ "$image_count" -lt "$MIN_FLICKR_IMAGES" ]]
}

needs_manifest_refresh() {
  local manifest_file_path="$1"
  local min_rows="$2"
  [[ ! -f "$manifest_file_path" ]] && return 0
  local row_count
  row_count="$(count_jsonl_rows "$manifest_file_path")"
  [[ "$row_count" -lt "$min_rows" ]]
}

inspect_private_adapt_set() {
  "$PYTHON_BIN_PATH" - <<PY
import json
from pathlib import Path

adapt_path = Path("${PRIVATE_DATA_DIR}/private_adapt_data.jsonl").expanduser().resolve()
gallery_root = Path("${PRIVATE_GALLERY_DIR}").expanduser().resolve()
threshold = float("${PRIVATE_ADAPT_MISSING_RETRAIN_THRESHOLD}")

if not adapt_path.exists():
    print(json.dumps({
        "action": "resample",
        "reason": "adaptation set missing",
        "tracked_rows": 0,
        "missing_rows": 0,
        "missing_ratio": 0.0,
        "threshold": threshold,
    }))
    raise SystemExit(0)

try:
    rows = [json.loads(line) for line in adapt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
except Exception as exc:
    print(json.dumps({
        "action": "resample",
        "reason": f"adaptation set unreadable: {exc}",
        "tracked_rows": 0,
        "missing_rows": 0,
        "missing_ratio": 0.0,
        "threshold": threshold,
    }, ensure_ascii=False))
    raise SystemExit(0)

tracked_rows = len(rows)
if tracked_rows == 0:
    print(json.dumps({
        "action": "resample",
        "reason": "adaptation set empty",
        "tracked_rows": 0,
        "missing_rows": 0,
        "missing_ratio": 0.0,
        "threshold": threshold,
    }))
    raise SystemExit(0)

missing_rows = 0
for row in rows:
    image_path = Path(row["image_path"]).expanduser().resolve()
    try:
        image_path.relative_to(gallery_root)
    except ValueError:
        missing_rows += 1
        continue
    if not image_path.is_file():
        missing_rows += 1

missing_ratio = missing_rows / tracked_rows
action = "resample" if missing_ratio >= threshold else "reuse"
reason = "missing ratio reached retrain threshold" if action == "resample" else "missing ratio below retrain threshold"
print(json.dumps({
    "action": action,
    "reason": reason,
    "tracked_rows": tracked_rows,
    "missing_rows": missing_rows,
    "missing_ratio": round(missing_ratio, 6),
    "threshold": threshold,
}))
PY
}

json_field() {
  local payload="$1"
  local field="$2"
  JSON_PAYLOAD="$payload" "$PYTHON_BIN_PATH" - "$field" <<'PY'
import json
import os
import sys

field = sys.argv[1]
payload = json.loads(os.environ["JSON_PAYLOAD"])
value = payload.get(field, "")
if isinstance(value, float):
    print(f"{value:.6f}")
else:
    print(value)
PY
}

if [[ "$PREPARE_PUBLIC_DATA" == "1" ]]; then
  log_step "Preparing public training data"
  if [[ "${FORCE:-0}" == "1" ]] || needs_flickr_refresh; then
      log_step "Refreshing Flickr30k"
      "$PYTHON_BIN_PATH" "$ROOT_DIR/tools/prepare_flickr30k.py" --output-dir "$ROOT_DIR/datasets/flickr30k"
  else
      log_step "Reusing Flickr30k"
      log_kv "captions_file_path=$ROOT_DIR/datasets/flickr30k/captions.txt"
  fi

  if [[ "${FORCE:-0}" == "1" ]] || needs_manifest_refresh "$ROOT_DIR/datasets/screen2words_train/manifest.jsonl" "$MIN_SCREEN2WORDS_TRAIN_ROWS"; then
      log_step "Refreshing Screen2Words train"
      "$PYTHON_BIN_PATH" "$ROOT_DIR/tools/prepare_screen2words.py" \
      --split train \
      --output-dir "$ROOT_DIR/datasets/screen2words_train"
  else
      log_step "Reusing Screen2Words train"
      log_kv "manifest_file_path=$ROOT_DIR/datasets/screen2words_train/manifest.jsonl"
  fi

  if [[ "${FORCE:-0}" == "1" ]] || needs_manifest_refresh "$ROOT_DIR/datasets/screen2words_val/manifest.jsonl" "$MIN_SCREEN2WORDS_VAL_ROWS"; then
      log_step "Refreshing Screen2Words val"
      "$PYTHON_BIN_PATH" "$ROOT_DIR/tools/prepare_screen2words.py" \
      --split val \
      --output-dir "$ROOT_DIR/datasets/screen2words_val"
  else
      log_step "Reusing Screen2Words val"
      log_kv "manifest_file_path=$ROOT_DIR/datasets/screen2words_val/manifest.jsonl"
  fi
else
  log_step "Skipping full public data download"
fi

if [[ -n "$PRIVATE_GALLERY_DIR" ]]; then
  require_dir "$PRIVATE_GALLERY_DIR"
  log_step "Building local gallery manifest"
  log_kv "private_gallery_dir=$(cd "$PRIVATE_GALLERY_DIR" && pwd)"

  gallery_cmd=(
    "$PYTHON_BIN_PATH" "$ROOT_DIR/tools/prepare_gallery_manifest.py"
    --gallery-path "$PRIVATE_GALLERY_DIR"
    --output-path "$PRIVATE_DATA_DIR/full_manifest.jsonl"
    --source-name private_gallery_local
  )
  "${gallery_cmd[@]}"

  adapt_status_json="$(inspect_private_adapt_set)"
  adapt_action="$(json_field "$adapt_status_json" action)"
  tracked_rows="$(json_field "$adapt_status_json" tracked_rows)"
  missing_rows="$(json_field "$adapt_status_json" missing_rows)"
  missing_ratio="$(json_field "$adapt_status_json" missing_ratio)"
  threshold="$(json_field "$adapt_status_json" threshold)"
  reason="$(json_field "$adapt_status_json" reason)"

  if [[ "$adapt_action" == "reuse" ]]; then
    log_step "Reusing existing private adaptation set"
    log_kv "tracked_rows=$tracked_rows"
    log_kv "missing_rows=$missing_rows"
    log_kv "missing_ratio=$missing_ratio"
    log_kv "retrain_threshold=$threshold"
  else
    log_step "Sampling capped private adaptation set"
    log_kv "target_size=$PRIVATE_ADAPT_TARGET_SIZE"
    log_kv "reason=$reason"
    if [[ "$tracked_rows" != "0" ]]; then
      log_kv "tracked_rows=$tracked_rows"
      log_kv "missing_rows=$missing_rows"
      log_kv "missing_ratio=$missing_ratio"
      log_kv "retrain_threshold=$threshold"
    fi
    adapt_cmd=(
      "$PYTHON_BIN_PATH" "$ROOT_DIR/tools/prepare_private_adapt_manifest.py"
      --source-manifest "$PRIVATE_DATA_DIR/full_manifest.jsonl"
      --target-size "$PRIVATE_ADAPT_TARGET_SIZE"
      --output-path "$PRIVATE_DATA_DIR/private_adapt_data.jsonl"
    )
    "${adapt_cmd[@]}"
  fi
elif [[ ! -f "$PRIVATE_DATA_DIR/full_manifest.jsonl" || ! -f "$PRIVATE_DATA_DIR/private_adapt_data.jsonl" ]]; then
  die "set PRIVATE_GALLERY_DIR to build the capped private adaptation set."
fi

printf 'data_ready=%s\n' "$ROOT_DIR/datasets"
