#!/usr/bin/env bash

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

GALLERY_DIR="${GALLERY_DIR:-${1:-}}"
FINAL_RUN_DIR="$ROOT_DIR/logs/semanticgallery_private_data_adapted"
FINAL_WEIGHTS_FILE_PATH="$FINAL_RUN_DIR/weights.safetensors"
QUICKSTART_STATE_FILE_PATH="$FINAL_RUN_DIR/quickstart_state.json"
PRIVATE_MANIFEST_FILE_PATH="$ROOT_DIR/datasets/private_gallery_local/private_adapt_data.jsonl"
STAGE1_WEIGHTS_FILE_PATH="${STAGE1_WEIGHTS_FILE_PATH:-$LOCAL_STAGE1_WEIGHTS_FILE_PATH}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-36168}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-$ROOT_DIR/deployment/search_config_gallery_mlx.json}"
METADATA_MANIFEST_FILE_PATH="${METADATA_MANIFEST_FILE_PATH:-$ROOT_DIR/datasets/private_gallery_local/full_manifest.jsonl}"
MODEL_PRECISION="${MODEL_PRECISION:-bfloat16}"
RUNTIME_DIR="${RUNTIME_DIR:-$ROOT_DIR/logs/runtime}"
LOG_FILE_PATH="${LOG_FILE_PATH:-$RUNTIME_DIR/semanticgallery_${PORT}.log}"
PID_FILE_PATH="${PID_FILE_PATH:-$RUNTIME_DIR/semanticgallery_${PORT}.pid}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-300}"

[[ -n "$GALLERY_DIR" ]] || die "set GALLERY_DIR to the folder you want to search."
require_dir "$GALLERY_DIR"

mkdir -p "$RUNTIME_DIR"

if [[ -f "$PID_FILE_PATH" ]]; then
  existing_pid="$(tr -d '[:space:]' < "$PID_FILE_PATH")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    die "SemanticGallery is already running on pid $existing_pid. Stop it first or use a different PORT."
  fi
  rm -f "$PID_FILE_PATH"
fi

ensure_port_free "$HOST" "$PORT"

: > "$LOG_FILE_PATH"

log_step() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_FILE_PATH" >&2
}

log_kv() {
  printf '  - %s\n' "$*" | tee -a "$LOG_FILE_PATH" >&2
}

die() {
  printf 'error: %s\n' "$*" | tee -a "$LOG_FILE_PATH" >&2
  exit 1
}

run_logged() {
  "$@" > >(tee -a "$LOG_FILE_PATH") 2> >(tee -a "$LOG_FILE_PATH" >&2)
}

run_logged ensure_env

resolved_gallery_dir="$(cd "$GALLERY_DIR" && pwd)"
resolved_stage1_weights_file_path="$(
  resolve_stage1_weights_file "$STAGE1_WEIGHTS_FILE_PATH" 2> >(tee -a "$LOG_FILE_PATH" >&2)
)"

log_step "Preparing adaptation data"
run_logged env PREPARE_PUBLIC_DATA=0 PRIVATE_GALLERY_DIR="$resolved_gallery_dir" "$ROOT_DIR/scripts/prepare_data.sh"

require_file "$PRIVATE_MANIFEST_FILE_PATH"

current_signature="$(
  "$PYTHON_BIN_PATH" - <<PY
import hashlib
import json
from pathlib import Path

gallery_dir = Path("${resolved_gallery_dir}").resolve().as_posix()
stage1_weights = Path("${resolved_stage1_weights_file_path}").resolve().as_posix()
manifest = Path("${PRIVATE_MANIFEST_FILE_PATH}").resolve()
payload = {
    "gallery_dir": gallery_dir,
    "stage1_weights_file_path": stage1_weights,
    "manifest_sha1": hashlib.sha1(manifest.read_bytes()).hexdigest(),
}
print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
PY
)"

stored_signature=""
if [[ -f "$QUICKSTART_STATE_FILE_PATH" ]]; then
  stored_signature="$(
    "$PYTHON_BIN_PATH" - <<PY
import json
from pathlib import Path

state_path = Path("${QUICKSTART_STATE_FILE_PATH}")
try:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
except Exception:
    payload = {}
print(payload.get("signature", ""))
PY
  )"
fi

if [[ ! -f "$FINAL_WEIGHTS_FILE_PATH" || "$current_signature" != "$stored_signature" ]]; then
  STAGE1_WEIGHTS_FILE_PATH="$resolved_stage1_weights_file_path" "$ROOT_DIR/scripts/adapt_best.sh"
  CURRENT_SIGNATURE="$current_signature" "$PYTHON_BIN_PATH" - <<PY
import json
import os
from pathlib import Path

state_path = Path("${QUICKSTART_STATE_FILE_PATH}")
state_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "signature": os.environ["CURRENT_SIGNATURE"],
    "gallery_dir": "${resolved_gallery_dir}",
    "stage1_weights_file_path": "${resolved_stage1_weights_file_path}",
    "final_weights_file_path": "${FINAL_WEIGHTS_FILE_PATH}",
}
state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
else
  log_step "Reusing existing Stage 2 adaptation"
  log_kv "weights_file_path=$FINAL_WEIGHTS_FILE_PATH"
fi

log_step "Starting SemanticGallery in background"
log_kv "host=$HOST"
log_kv "port=$PORT"
log_kv "gallery_dir=$resolved_gallery_dir"
log_kv "log_file_path=$LOG_FILE_PATH"
log_kv "pid_file_path=$PID_FILE_PATH"
"$PYTHON_BIN_PATH" - <<PY
import os
import subprocess
from pathlib import Path

root_dir = Path("${ROOT_DIR}")
log_path = Path("${LOG_FILE_PATH}")
pid_path = Path("${PID_FILE_PATH}")
log_path.parent.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env.update(
    {
        "PYTHONUNBUFFERED": "1",
        "GALLERY_DIR": "${resolved_gallery_dir}",
        "HOST": "${HOST}",
        "PORT": "${PORT}",
        "CONFIG_FILE_PATH": "${CONFIG_FILE_PATH}",
        "METADATA_MANIFEST_FILE_PATH": "${METADATA_MANIFEST_FILE_PATH}",
        "MODEL_PRECISION": "${MODEL_PRECISION}",
        "MODEL_WEIGHTS_FILE_PATH": "${FINAL_WEIGHTS_FILE_PATH}",
        "FORCE": "${FORCE:-0}",
        "ENCODE_BATCH_SIZE": "${ENCODE_BATCH_SIZE:-8}",
    }
)

with log_path.open("ab") as handle:
    proc = subprocess.Popen(
        ["/bin/bash", str(root_dir / "scripts" / "deploy_best.sh")],
        cwd=root_dir.as_posix(),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
PY

pid="$(tr -d '[:space:]' < "$PID_FILE_PATH")"
[[ -n "$pid" ]] || die "failed to record SemanticGallery pid"

log_step "Background job started"
log_kv "pid=$pid"
log_kv "url=http://$HOST:$PORT"
log_kv "startup_log=$LOG_FILE_PATH"

log_step "Following startup log"
"$PYTHON_BIN_PATH" - <<PY
from pathlib import Path
import os
import socket
import sys
import time

log_path = Path("${LOG_FILE_PATH}")
pid = int("${pid}")
idle_timeout_seconds = int("${STARTUP_TIMEOUT_SECONDS}")
ready_marker = "Uvicorn running on http://"
offset = log_path.stat().st_size if log_path.exists() else 0
last_progress_time = time.time()
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

while True:
    if log_path.exists():
        data = log_path.read_text(encoding="utf-8", errors="replace")
        if len(data) > offset:
            sys.stderr.write(data[offset:])
            sys.stderr.flush()
            offset = len(data)
            last_progress_time = time.time()
        if ready_marker in data:
            ready_logged = True

    if ready_logged and socket_ready("${HOST}", int("${PORT}")):
        ready_socket = True
        break

    if not pid_alive(pid):
        break
    if idle_timeout_seconds > 0 and time.time() - last_progress_time > idle_timeout_seconds:
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
    f"error: No new startup log output was observed for {idle_timeout_seconds}s. "
    f"The background job may be stalled. Check {log_path} or set a larger "
    f"STARTUP_TIMEOUT_SECONDS if your machine pauses for long stretches during startup.\n"
)
sys.exit(1)
PY
