#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MLX_MODEL_DIR="$ROOT/.cache/mlx/siglip2-base-patch16-224-f32"
LOCAL_STAGE1_WEIGHTS="$ROOT/logs/semanticgallery_public_stage1/weights.safetensors"
LOCAL_FINAL_WEIGHTS="$ROOT/logs/semanticgallery_private_data_adapted/weights.safetensors"
PUBLISHED_STAGE1_REPO_ID="Lucas20250626/semanticgallery-mlx-siglip2-stage1"
PUBLISHED_STAGE1_REVISION="main"
PUBLISHED_STAGE1_CACHE_DIR="$ROOT/.cache/semanticgallery/stage1"
PUBLISHED_STAGE1_WEIGHTS="$PUBLISHED_STAGE1_CACHE_DIR/weights.safetensors"
PUBLISHED_STAGE1_SUMMARY="$PUBLISHED_STAGE1_CACHE_DIR/summary.json"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_REPO_ID="Lucas20250626/semanticgallery-stage2-public-anchor"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_REVISION="main"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR="$ROOT/.cache/semanticgallery/stage2_public_anchor"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_ARCHIVE="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR/semanticgallery-stage2-public-anchor.tar.gz"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_METADATA="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR/sample_info.json"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR/extracted"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT/flickr30k"
PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST="$PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT/screen2words/manifest.jsonl"

timestamp() {
  date '+%H:%M:%S'
}

log_step() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >&2
}

log_kv() {
  printf '  - %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_uv() {
  command -v uv >/dev/null 2>&1 || die "uv is required."
}

ensure_env() {
  require_uv
  if [[ ! -x "$PY" ]]; then
    log_step "Creating Python environment"
    uv venv "$ROOT/.venv" --python 3.12
  fi
  if ! "$PY" - <<'PY' >/dev/null 2>&1; then
import datasets, fastapi, huggingface_hub, jinja2, mlx, mlx_embeddings, pillow_heif, tqdm, uvicorn
PY
    log_step "Installing Python dependencies"
    uv pip install --python "$PY" -r "$ROOT/requirements.txt"
  fi
}

ensure_mlx_model() {
  ensure_env
  if [[ ! -f "$MLX_MODEL_DIR/config.json" ]]; then
    log_step "Preparing MLX SigLIP2 model cache"
    "$PY" - <<PY
from pathlib import Path
from mlx_embeddings.convert import convert

root = Path("${ROOT}")
model_dir = root / ".cache" / "mlx" / "siglip2-base-patch16-224-f32"
model_dir.parent.mkdir(parents=True, exist_ok=True)
convert(
    "google/siglip2-base-patch16-224",
    mlx_path=model_dir.as_posix(),
    dtype="float32",
    skip_vision=False,
)
PY
  fi
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "missing directory: $path"
}

ensure_port_free() {
  local host="$1"
  local port="$2"
  if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $port is already in use on $host"
  fi
}

ensure_published_stage1_weights() {
  ensure_env
  if [[ ! -f "$PUBLISHED_STAGE1_WEIGHTS" ]]; then
    log_step "Downloading published Stage 1 weights"
    log_kv "repo_id=$PUBLISHED_STAGE1_REPO_ID"
    log_kv "files=weights.safetensors,summary.json"
    mkdir -p "$PUBLISHED_STAGE1_CACHE_DIR"
    "$PY" - <<PY
from pathlib import Path
import sys
from huggingface_hub import hf_hub_download

cache_dir = Path("${PUBLISHED_STAGE1_CACHE_DIR}")
cache_dir.mkdir(parents=True, exist_ok=True)
for filename in ("weights.safetensors", "summary.json"):
    print(f"downloading_stage1_file={filename}", file=sys.stderr)
    hf_hub_download(
        repo_id="${PUBLISHED_STAGE1_REPO_ID}",
        repo_type="model",
        revision="${PUBLISHED_STAGE1_REVISION}",
        filename=filename,
        local_dir=cache_dir.as_posix(),
    )
PY
  fi
}

ensure_published_stage2_public_anchor() {
  ensure_env
  if [[ -f "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_FLICKR_DIR/captions.txt" && -f "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_SCREEN2WORDS_MANIFEST" ]]; then
    log_step "Reusing cached Stage 2 public anchor"
    log_kv "cache_dir=$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR"
    return
  fi

  log_step "Downloading Stage 2 public anchor"
  log_kv "repo_id=$PUBLISHED_STAGE2_PUBLIC_ANCHOR_REPO_ID"
  log_kv "files=semanticgallery-stage2-public-anchor.tar.gz,sample_info.json"
  mkdir -p "$PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR"
  "$PY" - <<PY
import shutil
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

cache_dir = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_CACHE_DIR}")
archive_path = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_ARCHIVE}")
metadata_path = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_METADATA}")
extract_root = Path("${PUBLISHED_STAGE2_PUBLIC_ANCHOR_ROOT}")
tmp_root = cache_dir / "extracting"

cache_dir.mkdir(parents=True, exist_ok=True)
for filename in ("semanticgallery-stage2-public-anchor.tar.gz", "sample_info.json"):
    print(f"downloading_stage2_public_anchor_file={filename}", file=sys.stderr)
    hf_hub_download(
        repo_id="${PUBLISHED_STAGE2_PUBLIC_ANCHOR_REPO_ID}",
        repo_type="dataset",
        revision="${PUBLISHED_STAGE2_PUBLIC_ANCHOR_REVISION}",
        filename=filename,
        local_dir=cache_dir.as_posix(),
    )

if tmp_root.exists():
    shutil.rmtree(tmp_root)
tmp_root.mkdir(parents=True, exist_ok=True)
print("extracting_stage2_public_anchor=true", file=sys.stderr)
with tarfile.open(archive_path, "r:gz") as tar:
    try:
        tar.extractall(tmp_root, filter="data")
    except TypeError:
        tar.extractall(tmp_root)

if extract_root.exists():
    shutil.rmtree(extract_root)
tmp_root.rename(extract_root)

if metadata_path.exists():
    shutil.copy2(metadata_path, extract_root / "sample_info.json")
print("stage2_public_anchor_ready=true", file=sys.stderr)
PY
}

resolve_stage1_weights_file() {
  local explicit="${1:-}"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return
  fi
  if [[ -f "$LOCAL_STAGE1_WEIGHTS" ]]; then
    printf '%s\n' "$LOCAL_STAGE1_WEIGHTS"
    return
  fi
  if [[ -f "$PUBLISHED_STAGE1_WEIGHTS" ]]; then
    printf '%s\n' "$PUBLISHED_STAGE1_WEIGHTS"
    return
  fi
  ensure_published_stage1_weights
  printf '%s\n' "$PUBLISHED_STAGE1_WEIGHTS"
}

resolve_weights_file() {
  local explicit="${1:-}"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return
  fi
  if [[ -f "$LOCAL_FINAL_WEIGHTS" ]]; then
    printf '%s\n' "$LOCAL_FINAL_WEIGHTS"
    return
  fi
  printf '%s\n' "$(resolve_stage1_weights_file)"
}

prepare_mlx_search_config() {
  local gallery_path="$1"
  local config_output="$2"
  local metadata_manifest="${3:-}"
  local weights_file="${4:-}"
  local precision="${5:-bfloat16}"
  local gallery_name
  local embeddings_output
  local paths_output
  local skipped_output
  local legacy_embeddings_output
  local legacy_paths_output
  local legacy_skipped_output
  local encode_cmd
  local create_cmd

  ensure_mlx_model

  gallery_path="$(cd "$gallery_path" && pwd)"
  gallery_name="$(basename "$gallery_path")"
  embeddings_output="$ROOT/deployment/${gallery_name}_mlx_siglip2_embeddings.npy"
  paths_output="$ROOT/deployment/${gallery_name}_mlx_siglip2.paths.txt"
  skipped_output="$ROOT/deployment/${gallery_name}_mlx_siglip2_skipped.json"
  legacy_embeddings_output="$ROOT/deployment/${gallery_name}_siglip2_embeddings.npy"
  legacy_paths_output="$ROOT/deployment/${gallery_name}_siglip2.paths.txt"
  legacy_skipped_output="$ROOT/deployment/${gallery_name}_siglip2_skipped.json"

  if [[ "${FORCE:-0}" != "1" && -z "$weights_file" && -f "$legacy_embeddings_output" && -f "$legacy_paths_output" && -f "$legacy_skipped_output" ]]; then
    embeddings_output="$legacy_embeddings_output"
    paths_output="$legacy_paths_output"
    skipped_output="$legacy_skipped_output"
  fi

  log_step "Preparing gallery bank"
  log_kv "gallery_path=$gallery_path"
  log_kv "precision=$precision"
  log_kv "model_path=$MLX_MODEL_DIR"
  if [[ -n "$weights_file" && -f "$weights_file" ]]; then
    log_kv "weights_file=$weights_file"
  else
    log_kv "weights_file=none"
  fi

  if [[ "${FORCE:-0}" == "1" || ! -f "$embeddings_output" || ! -f "$paths_output" || ! -f "$skipped_output" ]]; then
    log_step "Encoding gallery images"
    encode_cmd=(
      "$PY" "$ROOT/deployment/encode_gallery.py"
      --gallery-path "$gallery_path"
      --model-path "$MLX_MODEL_DIR"
      --precision "$precision"
      --batch-size "${ENCODE_BATCH_SIZE:-8}"
      --embeddings-output "$embeddings_output"
      --paths-output "$paths_output"
      --skipped-output "$skipped_output"
    )
    if [[ -n "$weights_file" && -f "$weights_file" ]]; then
      encode_cmd+=(--weights-file "$weights_file")
    fi
    "${encode_cmd[@]}"
  else
    log_step "Reusing existing gallery bank"
    log_kv "embeddings_file=$embeddings_output"
    log_kv "indexed_paths_file=$paths_output"
    log_kv "skipped_images_file=$skipped_output"
  fi

  log_step "Writing search config"
  create_cmd=(
    "$PY" "$ROOT/deployment/create_index.py"
    --gallery-path "$gallery_path"
    --model-path "$MLX_MODEL_DIR"
    --precision "$precision"
    --embeddings-file "$embeddings_output"
    --indexed-paths-file "$paths_output"
    --skipped-file "$skipped_output"
    --config-output "$config_output"
  )
  if [[ -n "$weights_file" && -f "$weights_file" ]]; then
    create_cmd+=(--weights-file "$weights_file")
  fi
  if [[ -n "$metadata_manifest" && -f "$metadata_manifest" ]]; then
    create_cmd+=(--metadata-manifest "$metadata_manifest")
  fi
  "${create_cmd[@]}"
}
