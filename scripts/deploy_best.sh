#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GALLERY_PATH="${GALLERY_PATH:-${1:-}}"
CONFIG_OUTPUT="${CONFIG_OUTPUT:-$ROOT/deployment/search_config_gallery_mlx.json}"
METADATA_MANIFEST="${METADATA_MANIFEST:-$ROOT/datasets/private_gallery_local/full_manifest.jsonl}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36168}"
MODEL_PRECISION="${MODEL_PRECISION:-bfloat16}"
MODEL_WEIGHTS="${MODEL_WEIGHTS:-}"

[[ -n "$GALLERY_PATH" ]] || die "set GALLERY_PATH to the folder you want to search."
require_dir "$GALLERY_PATH"
ensure_port_free "$HOST" "$PORT"

log_step "SemanticGallery startup"
log_kv "host=$HOST"
log_kv "port=$PORT"
log_kv "gallery_path=$(cd "$GALLERY_PATH" && pwd)"

RESOLVED_WEIGHTS="$(resolve_weights_file "$MODEL_WEIGHTS")"
prepare_mlx_search_config "$GALLERY_PATH" "$CONFIG_OUTPUT" "$METADATA_MANIFEST" "$RESOLVED_WEIGHTS" "$MODEL_PRECISION"

log_step "Launching web app"
log_kv "url=http://$HOST:$PORT"
log_kv "config_output=$CONFIG_OUTPUT"

exec "$PY" "$ROOT/deployment/web_app.py" \
  --config "$CONFIG_OUTPUT" \
  --host "$HOST" \
  --port "$PORT"
