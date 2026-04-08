#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GALLERY_DIR="${GALLERY_DIR:-${1:-}}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-}"
METADATA_MANIFEST_FILE_PATH="${METADATA_MANIFEST_FILE_PATH:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36168}"
MODEL_PRECISION="${MODEL_PRECISION:-bfloat16}"
MODEL_WEIGHTS_FILE_PATH="${MODEL_WEIGHTS_FILE_PATH:-}"

[[ -n "$GALLERY_DIR" ]] || die "set GALLERY_DIR to the folder you want to search."
require_dir "$GALLERY_DIR"
ensure_port_free "$HOST" "$PORT"

RESOLVED_GALLERY_DIR="$(cd "$GALLERY_DIR" && pwd)"
GALLERY_KEY="$(gallery_artifact_key "$RESOLVED_GALLERY_DIR")"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$ROOT_DIR/deployment/search_configs/${GALLERY_KEY}.json}"
PRIVATE_DATA_DIR="${PRIVATE_DATA_DIR:-$ROOT_DIR/datasets/private_gallery_local/$GALLERY_KEY}"
METADATA_MANIFEST_FILE_PATH="${METADATA_MANIFEST_FILE_PATH:-$PRIVATE_DATA_DIR/full_manifest.jsonl}"
MODEL_WEIGHTS_FILE_PATH="${MODEL_WEIGHTS_FILE_PATH:-$ROOT_DIR/logs/semanticgallery_private_data_adapted/${GALLERY_KEY}/weights.safetensors}"

log_step "SemanticGallery startup"
log_kv "host=$HOST"
log_kv "port=$PORT"
log_kv "gallery_dir=$RESOLVED_GALLERY_DIR"

RESOLVED_WEIGHTS_FILE_PATH="$(resolve_weights_file "$MODEL_WEIGHTS_FILE_PATH")"
prepare_mlx_search_config "$RESOLVED_GALLERY_DIR" "$CONFIG_FILE_PATH" "$METADATA_MANIFEST_FILE_PATH" "$RESOLVED_WEIGHTS_FILE_PATH" "$MODEL_PRECISION"

log_step "Launching web app"
log_kv "url=http://$HOST:$PORT"
log_kv "config_file_path=$CONFIG_FILE_PATH"

exec "$PYTHON_BIN_PATH" "$ROOT_DIR/deployment/web_app.py" \
  --config "$CONFIG_FILE_PATH" \
  --host "$HOST" \
  --port "$PORT"
