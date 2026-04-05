#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GALLERY_PATH="${GALLERY_PATH:-${1:-}}"
FINAL_RUN_DIR="$ROOT/logs/semanticgallery_private_data_adapted"
FINAL_WEIGHTS="$FINAL_RUN_DIR/weights.safetensors"
QUICKSTART_STATE_FILE="$FINAL_RUN_DIR/quickstart_state.json"
PRIVATE_MANIFEST="$ROOT/datasets/private_gallery_local/private_adapt_data.jsonl"
STAGE1_WEIGHTS="${STAGE1_WEIGHTS:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36168}"
CONFIG_OUTPUT="${CONFIG_OUTPUT:-$ROOT/deployment/search_config_gallery_mlx.json}"
METADATA_MANIFEST="${METADATA_MANIFEST:-$ROOT/datasets/private_gallery_local/full_manifest.jsonl}"
MODEL_PRECISION="${MODEL_PRECISION:-bfloat16}"
RUNTIME_DIR="${RUNTIME_DIR:-$ROOT/logs/runtime}"
LOG_FILE="${LOG_FILE:-$RUNTIME_DIR/semanticgallery_${PORT}.log}"
PID_FILE="${PID_FILE:-$RUNTIME_DIR/semanticgallery_${PORT}.pid}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-300}"

[[ -n "$GALLERY_PATH" ]] || die "set GALLERY_PATH to the folder you want to search."
require_dir "$GALLERY_PATH"

ensure_env

resolved_gallery_path="$(cd "$GALLERY_PATH" && pwd)"
resolved_stage1_weights="$(resolve_stage1_weights_file "$STAGE1_WEIGHTS")"

log_step "Preparing adaptation data"
PREPARE_PUBLIC_DATA=0 PRIVATE_GALLERY_PATH="$resolved_gallery_path" "$ROOT/scripts/prepare_data.sh"

require_file "$PRIVATE_MANIFEST"

current_signature="$(
  "$PY" - <<PY
import hashlib
import json
from pathlib import Path

gallery_path = Path("${resolved_gallery_path}").resolve().as_posix()
stage1_weights = Path("${resolved_stage1_weights}").resolve().as_posix()
manifest = Path("${PRIVATE_MANIFEST}").resolve()
payload = {
    "gallery_path": gallery_path,
    "stage1_weights": stage1_weights,
    "manifest_sha1": hashlib.sha1(manifest.read_bytes()).hexdigest(),
}
print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
PY
)"

stored_signature=""
if [[ -f "$QUICKSTART_STATE_FILE" ]]; then
  stored_signature="$(
    "$PY" - <<PY
import json
from pathlib import Path

state_path = Path("${QUICKSTART_STATE_FILE}")
try:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    payload = {}
print(payload.get("signature", ""))
PY
  )"
fi

if [[ ! -f "$FINAL_WEIGHTS" || "$current_signature" != "$stored_signature" ]]; then
  STAGE1_WEIGHTS="$resolved_stage1_weights" "$ROOT/scripts/adapt_best.sh"
  CURRENT_SIGNATURE="$current_signature" "$PY" - <<PY
import json
import os
from pathlib import Path

state_path = Path("${QUICKSTART_STATE_FILE}")
state_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "signature": os.environ["CURRENT_SIGNATURE"],
    "gallery_path": "${resolved_gallery_path}",
    "stage1_weights": "${resolved_stage1_weights}",
    "final_weights": "${FINAL_WEIGHTS}",
}
state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
else
  log_step "Reusing existing Stage 2 adaptation"
  log_kv "weights_file=$FINAL_WEIGHTS"
fi

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    die "SemanticGallery is already running on pid $existing_pid. Stop it first or use a different PORT."
  fi
  rm -f "$PID_FILE"
fi

log_step "Starting SemanticGallery in background"
log_kv "host=$HOST"
log_kv "port=$PORT"
log_kv "gallery_path=$resolved_gallery_path"
log_kv "log_file=$LOG_FILE"
log_kv "pid_file=$PID_FILE"
"$PY" - <<PY
import os
import subprocess
from pathlib import Path

root = Path("${ROOT}")
log_path = Path("${LOG_FILE}")
pid_path = Path("${PID_FILE}")
log_path.parent.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env.update(
    {
        "PYTHONUNBUFFERED": "1",
        "GALLERY_PATH": "${resolved_gallery_path}",
        "HOST": "${HOST}",
        "PORT": "${PORT}",
        "CONFIG_OUTPUT": "${CONFIG_OUTPUT}",
        "METADATA_MANIFEST": "${METADATA_MANIFEST}",
        "MODEL_PRECISION": "${MODEL_PRECISION}",
        "MODEL_WEIGHTS": "${FINAL_WEIGHTS}",
        "FORCE": "${FORCE:-0}",
        "ENCODE_BATCH_SIZE": "${ENCODE_BATCH_SIZE:-8}",
    }
)

with log_path.open("wb") as handle:
    proc = subprocess.Popen(
        ["/bin/bash", str(root / "scripts" / "deploy_best.sh")],
        cwd=root.as_posix(),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
PY

pid="$(tr -d '[:space:]' < "$PID_FILE")"
[[ -n "$pid" ]] || die "failed to record SemanticGallery pid"

log_step "Background job started"
log_kv "pid=$pid"
log_kv "url=http://$HOST:$PORT"
log_kv "startup_log=$LOG_FILE"

log_step "Following startup log"
"$PY" - <<PY
from pathlib import Path
import os
import socket
import sys
import time

log_path = Path("${LOG_FILE}")
pid = int("${pid}")
timeout_seconds = int("${STARTUP_TIMEOUT_SECONDS}")
ready_marker = "Uvicorn running on http://"
offset = 0
deadline = time.time() + timeout_seconds
ready_logged = False
ready_socket = False

def pid_alive(target: int) -> bool:
    try:
        os.kill(target, 0)
    except OSError:
        return False
    return True

def socket_ready(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0

while time.time() < deadline:
    if log_path.exists():
        data = log_path.read_text(encoding="utf-8", errors="replace")
        if len(data) > offset:
            sys.stderr.write(data[offset:])
            sys.stderr.flush()
            offset = len(data)
        if ready_marker in data:
            ready_logged = True

    if ready_logged and socket_ready("${HOST}", int("${PORT}")):
        ready_socket = True
        break

    if not pid_alive(pid):
        break
    time.sleep(0.25)

if log_path.exists():
    data = log_path.read_text(encoding="utf-8", errors="replace")
    if len(data) > offset:
        sys.stderr.write(data[offset:])
        sys.stderr.flush()

if ready_logged and ready_socket:
    current_time = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[{current_time}] SemanticGallery is ready at http://${HOST}:${PORT}\n")
    sys.exit(0)

if not pid_alive(pid):
    sys.stderr.write(f"error: SemanticGallery exited during startup. See {log_path}\n")
    sys.exit(1)

sys.stderr.write(
    f"error: Timed out after {timeout_seconds}s waiting for SemanticGallery to report readiness. "
    f"See {log_path}\n"
)
sys.exit(1)
PY
