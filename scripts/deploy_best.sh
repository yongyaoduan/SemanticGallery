#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GALLERY_DIR="${GALLERY_DIR:-${1:-}}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$ROOT_DIR/deployment/search_config_gallery_mlx.json}"
METADATA_MANIFEST_FILE_PATH="${METADATA_MANIFEST_FILE_PATH:-$ROOT_DIR/datasets/private_gallery_local/full_manifest.jsonl}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36168}"
MODEL_PRECISION="${MODEL_PRECISION:-bfloat16}"
MODEL_WEIGHTS_FILE_PATH="${MODEL_WEIGHTS_FILE_PATH:-$LOCAL_FINAL_WEIGHTS_FILE_PATH}"

[[ -n "$GALLERY_DIR" ]] || die "set GALLERY_DIR to the folder you want to search."
require_dir "$GALLERY_DIR"
ensure_port_free "$HOST" "$PORT"

log_step "SemanticGallery startup"
log_kv "host=$HOST"
log_kv "port=$PORT"
log_kv "gallery_dir=$(cd "$GALLERY_DIR" && pwd)"

RESOLVED_WEIGHTS_FILE_PATH="$(resolve_weights_file "$MODEL_WEIGHTS_FILE_PATH")"
prepare_mlx_search_config "$GALLERY_DIR" "$CONFIG_FILE_PATH" "$METADATA_MANIFEST_FILE_PATH" "$RESOLVED_WEIGHTS_FILE_PATH" "$MODEL_PRECISION"

log_step "Launching web app"
log_kv "url=http://$HOST:$PORT"
log_kv "config_file_path=$CONFIG_FILE_PATH"

exec "$PYTHON_BIN_PATH" "$ROOT_DIR/deployment/web_app.py" \
  --config "$CONFIG_FILE_PATH" \
  --host "$HOST" \
  --port "$PORT"
